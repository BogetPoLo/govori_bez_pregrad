package ru.obucheniesluzheniyu.app.recognition

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import org.json.JSONObject
import ru.obucheniesluzheniyu.app.phrases.SignEntry
import java.io.Closeable
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.nio.FloatBuffer

internal class SlovoModel private constructor(
    private val session: OrtSession,
    val recipe: FrameRecipe,
    private val entries: List<SignEntry>,
) : Closeable {
    fun recognize(frames: List<GestureFrame>): SignEntry = recognizeTensor(recipe.tensor(frames))

    fun recognizeTensor(values: FloatBuffer): SignEntry = bestCatalogGesture(scores(values), entries)

    fun scores(values: FloatBuffer): FloatArray {
        OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), values, recipe.shape).use { input ->
            session.run(mapOf(session.inputNames.single() to input)).use { result ->
                val output = (result[0] as OnnxTensor).floatBuffer
                check(output.remaining() == CLASS_COUNT)
                return FloatArray(CLASS_COUNT) { output.get().also { score -> check(score.isFinite()) } }
            }
        }
    }

    override fun close() = session.close()

    companion object {
        private const val DIRECTORY = "recognition"
        private const val MODEL = "mvit32-2.onnx"
        private const val CLASS_COUNT = 1001
        private const val MAX_MODEL_BYTES = 200 * 1024 * 1024L
        private const val MODEL_URL =
            "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/mvit/onnx/mvit32-2.onnx"

        suspend fun open(context: Context, entries: List<SignEntry>, onDownload: () -> Unit = {}): SlovoModel {
            require(entries.isNotEmpty() && entries.all { it.classIndex in 0 until (CLASS_COUNT - 1) })
            require(entries.distinctBy { it.classIndex }.size == entries.size)
            val assets = context.assets
            val manifest = assets.open("$DIRECTORY/manifest.json").bufferedReader().use { JSONObject(it.readText()) }
            require(manifest.getString("recipe") == "rgb-letterbox-bilinear-uniform-v1")
            val recipe = FrameRecipe.from(manifest)
            val directory = File(context.noBackupFilesDir, DIRECTORY)
            check(directory.isDirectory || directory.mkdirs())
            val modelFile = File(directory, MODEL)
            if (modelFile.isFile) {
                val cachedSession = try {
                    openSession(modelFile, recipe.shape)
                } catch (_: OrtException) {
                    null
                } catch (_: IllegalArgumentException) {
                    null
                }
                if (cachedSession != null) return SlovoModel(cachedSession, recipe, entries)
                if (!modelFile.delete()) throw IOException("Cannot replace the cached Slovo model")
            }

            val temporary = File.createTempFile("mvit32-", ".onnx", directory)
            try {
                if (assets.list(DIRECTORY).orEmpty().contains(MODEL)) {
                    assets.open("$DIRECTORY/$MODEL").use { input ->
                        temporary.outputStream().use { output -> input.copyTo(output) }
                    }
                } else {
                    onDownload()
                    download(temporary)
                }
                currentCoroutineContext().ensureActive()
                val session = openSession(temporary, recipe.shape)
                if (!temporary.renameTo(modelFile)) {
                    session.close()
                    throw IOException("Cannot save the downloaded Slovo model")
                }
                return SlovoModel(session, recipe, entries)
            } finally {
                temporary.delete()
            }
        }

        private suspend fun download(destination: File) {
            val connection = URL(MODEL_URL).openConnection() as HttpURLConnection
            connection.connectTimeout = 15_000
            connection.readTimeout = 30_000
            connection.instanceFollowRedirects = false
            try {
                if (connection.responseCode != HttpURLConnection.HTTP_OK) {
                    throw IOException("Slovo download returned HTTP ${connection.responseCode}")
                }
                val expectedSize = connection.contentLengthLong
                if (expectedSize > MAX_MODEL_BYTES) throw IOException("Unexpected Slovo model size")
                var copied = 0L
                connection.inputStream.buffered().use { input ->
                    destination.outputStream().buffered().use { output ->
                        val buffer = ByteArray(64 * 1024)
                        while (true) {
                            currentCoroutineContext().ensureActive()
                            val count = input.read(buffer)
                            if (count == -1) break
                            copied += count
                            if (copied > MAX_MODEL_BYTES) throw IOException("Unexpected Slovo model size")
                            output.write(buffer, 0, count)
                        }
                    }
                }
                if (copied == 0L || (expectedSize >= 0 && copied != expectedSize)) {
                    throw IOException("Incomplete Slovo model download")
                }
            } finally {
                connection.disconnect()
            }
        }

        private fun openSession(file: File, shape: LongArray): OrtSession {
            val session = OrtSession.SessionOptions().use { options ->
                options.setIntraOpNumThreads(2)
                options.setCPUArenaAllocator(false)
                options.setMemoryPatternOptimization(false)
                OrtEnvironment.getEnvironment().createSession(file.absolutePath, options)
            }
            try {
                require((session.inputInfo.values.single().info as TensorInfo).matches(shape))
                val output = session.outputInfo.values.single().info as TensorInfo
                require(output.matches(longArrayOf(1, CLASS_COUNT.toLong())) ||
                    output.matches(longArrayOf(CLASS_COUNT.toLong())))
                return session
            } catch (error: Exception) {
                session.close()
                throw error
            }
        }

        private fun TensorInfo.matches(expected: LongArray): Boolean =
            type == OnnxJavaType.FLOAT && shape.size == expected.size &&
                shape.indices.all { shape[it] < 0 || shape[it] == expected[it] }
    }
}

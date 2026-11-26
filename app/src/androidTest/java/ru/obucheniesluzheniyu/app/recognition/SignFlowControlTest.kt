package ru.obucheniesluzheniyu.app.recognition

import android.content.res.AssetManager
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import ru.obucheniesluzheniyu.app.phrases.loadSignEntries
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

@RunWith(AndroidJUnit4::class)
class SignFlowControlTest {
    @Test
    fun desktopTensorProducesTheSameLabelOnAndroid() = runBlocking(Dispatchers.IO) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val assets = instrumentation.context.assets
        assumeTrue("Supply the expected label in the control assets", assets.list("").orEmpty().contains("mvit-expected.txt"))
        val expected = assets.open("mvit-expected.txt").bufferedReader().use { it.readText().trim() }
        val context = instrumentation.targetContext
        SlovoModel.open(context, loadSignEntries(context.assets)).use { model ->
            assertEquals(expected, model.recognizeTensor(assets.controlTensor("mvit-control.f32")).text)
        }
    }

    @Test
    fun desktopTensorProducesTheSameScoresOnAndroid() = runBlocking(Dispatchers.IO) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val assets = instrumentation.context.assets
        assumeTrue("Supply desktop scores in the control assets", assets.list("").orEmpty().contains("mvit-scores.json"))
        val context = instrumentation.targetContext
        val expected = assets.open("mvit-scores.json").bufferedReader().use { JSONArray(it.readText()) }
        SlovoModel.open(context, loadSignEntries(context.assets)).use { model ->
            val actual = model.scores(assets.controlTensor("mvit-control.f32"))
            assertEquals(expected.length(), actual.size)
            actual.forEachIndexed { index, score ->
                assertEquals("Score at index $index", expected.getDouble(index), score.toDouble(), 0.001)
            }
        }
    }
}

internal fun AssetManager.controlTensor(name: String): FloatBuffer {
    val input = open(name).use { it.readBytes() }
    val tensor = ByteBuffer.allocateDirect(input.size).order(ByteOrder.nativeOrder())
    val source = ByteBuffer.wrap(input).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
    return tensor.asFloatBuffer().apply {
        put(source)
        flip()
    }
}

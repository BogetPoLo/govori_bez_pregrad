package ru.obucheniesluzheniyu.app.recognition

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import androidx.camera.core.ImageProxy
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.roundToInt

internal data class GestureFrame(val timestampNs: Long, val pixels: IntArray)

internal class FrameRecipe private constructor(
    val frameCount: Int,
    private val width: Int,
    private val height: Int,
    private val layout: String,
    private val mean: FloatArray,
    private val std: FloatArray,
) {
    private val channelsFirst = layout != "NTCHW"

    val shape: LongArray
        get() = when (layout) {
            "N1CTHW" -> longArrayOf(1, 1, 3, frameCount.toLong(), height.toLong(), width.toLong())
            "NCTHW" -> longArrayOf(1, 3, frameCount.toLong(), height.toLong(), width.toLong())
            else -> longArrayOf(1, frameCount.toLong(), 3, height.toLong(), width.toLong())
        }

    fun frame(image: ImageProxy, timestampNs: Long): GestureFrame {
        val source = image.toBitmap()
        val output = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        try {
            val crop = image.cropRect
            val rotation = image.imageInfo.rotationDegrees
            val sideways = rotation == 90 || rotation == 270
            val rotatedWidth = if (sideways) crop.height() else crop.width()
            val rotatedHeight = if (sideways) crop.width() else crop.height()
            val scale = minOf(width.toFloat() / rotatedWidth, height.toFloat() / rotatedHeight)
            val scaledWidth = (rotatedWidth * scale).roundToInt().coerceIn(1, width)
            val scaledHeight = (rotatedHeight * scale).roundToInt().coerceIn(1, height)
            val left = (width - scaledWidth) / 2
            val top = (height - scaledHeight) / 2
            val transform = Matrix().apply {
                setTranslate(-crop.exactCenterX(), -crop.exactCenterY())
                postRotate(rotation.toFloat())
                postScale(scaledWidth.toFloat() / rotatedWidth, scaledHeight.toFloat() / rotatedHeight)
                postTranslate(left + scaledWidth / 2f, top + scaledHeight / 2f)
            }
            Canvas(output).apply {
                drawRGB(114, 114, 114)
                clipRect(left, top, left + scaledWidth, top + scaledHeight)
                drawBitmap(source, transform, Paint(Paint.FILTER_BITMAP_FLAG))
            }
            val pixels = IntArray(width * height)
            output.getPixels(pixels, 0, width, 0, 0, width, height)
            return GestureFrame(timestampNs, pixels)
        } finally {
            output.recycle()
            source.recycle()
        }
    }

    fun tensor(frames: List<GestureFrame>): FloatBuffer {
        require(frames.isNotEmpty())
        val area = width * height
        val tensor = ByteBuffer.allocateDirect(frameCount * 3 * area * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        val start = frames.first().timestampNs
        val span = frames.last().timestampNs - start
        val selected = List(frameCount) { index ->
            val target = start + span * index / (frameCount - 1).coerceAtLeast(1)
            frames.minBy { abs(it.timestampNs - target) }.pixels
        }
        repeat(3 * frameCount) { plane ->
            val channel = if (channelsFirst) plane / frameCount else plane % 3
            val time = if (channelsFirst) plane % frameCount else plane / 3
            val shift = (2 - channel) * 8
            for (pixel in selected[time]) {
                val intensity = (pixel ushr shift) and 0xff
                tensor.put((intensity - mean[channel]) / std[channel])
            }
        }
        tensor.flip()
        return tensor
    }

    companion object {
        fun from(manifest: JSONObject): FrameRecipe {
            val frames = manifest.getInt("frames")
            val width = manifest.getInt("width")
            val height = manifest.getInt("height")
            val layout = manifest.getString("layout")
            require(frames in 1..64)
            require(width in 1..320 && height in 1..320)
            require(layout == "N1CTHW" || layout == "NCTHW" || layout == "NTCHW")
            return FrameRecipe(
                frames, width, height, layout,
                manifest.rgb("mean"), manifest.rgb("std").also { require(it.all { value -> value > 0f }) },
            )
        }

        private fun JSONObject.rgb(name: String): FloatArray {
            val values = getJSONArray(name)
            require(values.length() == 3)
            return FloatArray(3) { values.getDouble(it).toFloat().also { value -> require(value.isFinite()) } }
        }
    }
}

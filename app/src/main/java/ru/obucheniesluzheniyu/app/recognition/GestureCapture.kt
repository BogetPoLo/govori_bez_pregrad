package ru.obucheniesluzheniyu.app.recognition

import android.os.SystemClock
import androidx.camera.core.ImageProxy

internal class GestureCapture {
    private val lock = Any()
    private var recording: GestureFrames? = null

    fun start() = synchronized(lock) {
        recording = GestureFrames()
    }

    fun finish(stoppedAtNs: Long): List<GestureFrame> = synchronized(lock) {
        val frames = recording?.beforeStop(stoppedAtNs).orEmpty()
        recording = null
        frames
    }

    fun cancel() = synchronized(lock) {
        recording = null
    }

    fun analyze(image: ImageProxy, recipe: FrameRecipe) {
        // Use the same monotonic clock for frames and the Stop event.
        val timestampNs = SystemClock.elapsedRealtimeNanos()
        val current = synchronized(lock) {
            recording?.takeIf { it.needsFrame(timestampNs) }
        } ?: return

        val frame = recipe.frame(image, timestampNs)
        synchronized(lock) {
            if (recording === current) current.add(frame)
        }
    }
}

internal class GestureFrames {
    private val frames = ArrayList<GestureFrame>(MAX_FRAMES)
    private var intervalNs = 30_000_000L

    fun needsFrame(timestampNs: Long): Boolean =
        frames.lastOrNull()?.let { timestampNs - it.timestampNs >= intervalNs } ?: true

    fun add(frame: GestureFrame) {
        if (frames.size == MAX_FRAMES) {
            // Keep samples across the entire take instead of stopping at a full buffer.
            for (index in 1 until MAX_FRAMES / 2) {
                frames[index] = frames[index * 2]
            }
            frames.subList(MAX_FRAMES / 2, frames.size).clear()
            intervalNs *= 2
        }
        frames.add(frame)
    }

    fun beforeStop(stoppedAtNs: Long): List<GestureFrame> =
        frames.takeWhile { it.timestampNs <= stoppedAtNs - TRIM_NS }

    private companion object {
        const val MAX_FRAMES = 128
        const val TRIM_NS = 1_000_000_000L
    }
}

package ru.obucheniesluzheniyu.app.recognition

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class GestureFramesTest {
    @Test
    fun trimUsesStopTimeEvenWhenCameraHasAlreadyStoppedSendingFrames() {
        val frames = GestureFrames()
        listOf(0L, 900L, 1_000L, 1_100L, 1_800L).forEach {
            frames.add(frameAt(it))
        }

        assertEquals(
            listOf(0L, 900L, 1_000L),
            frames.beforeStop(2_000L * NS_PER_MS).map { it.timestampNs / NS_PER_MS },
        )
    }

    @Test
    fun noInputRemainsWhenStopComesBeforeTrimWindow() {
        val frames = GestureFrames()
        frames.add(frameAt(100L))

        assertTrue(frames.beforeStop(700L * NS_PER_MS).isEmpty())
    }

    @Test
    fun longRecordingRetainsSamplesFromStartAndEndWithBoundedMemory() {
        val frames = GestureFrames()
        for (timeMs in 0L..60_000L step 33L) {
            if (frames.needsFrame(timeMs * NS_PER_MS)) frames.add(frameAt(timeMs))
        }
        val retained = frames.beforeStop(60_000L * NS_PER_MS)
            .map { it.timestampNs / NS_PER_MS }

        assertEquals(0L, retained.first())
        assertTrue(retained.last() in 57_000L..59_000L)
        assertTrue(retained.size in 32..128)
        assertTrue(retained.zipWithNext().all { (previous, next) -> previous < next })
    }

    private fun frameAt(timeMs: Long) = GestureFrame(timeMs * NS_PER_MS, intArrayOf())

    private companion object {
        const val NS_PER_MS = 1_000_000L
    }
}

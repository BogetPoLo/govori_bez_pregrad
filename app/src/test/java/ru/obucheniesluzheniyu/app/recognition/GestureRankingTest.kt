package ru.obucheniesluzheniyu.app.recognition

import org.junit.Assert.assertSame
import org.junit.Test
import ru.obucheniesluzheniyu.app.phrases.SignEntry

class GestureRankingTest {
    private val hello = SignEntry("Привет", "https://signflow.ru/privet", null, 476)
    private val salt = SignEntry("Соль", "https://signflow.ru/sol", null, 775)

    @Test
    fun lowScoresReturnTheHighestScoringCatalogEntry() {
        val scores = FloatArray(1001)
        scores[hello.classIndex] = 0.01f
        scores[salt.classIndex] = 0.02f
        scores[1000] = 0.96f

        assertSame(salt, bestCatalogGesture(scores, listOf(hello, salt)))
    }

    @Test
    fun equallyScoredWordsFollowCatalogOrder() {
        assertSame(hello, bestCatalogGesture(FloatArray(1001), listOf(hello, salt)))
    }

    @Test
    fun rankingAlsoWorksWithNegativeScores() {
        val scores = FloatArray(1001)
        scores[hello.classIndex] = -2f
        scores[salt.classIndex] = -3f

        assertSame(hello, bestCatalogGesture(scores, listOf(salt, hello)))
    }
}

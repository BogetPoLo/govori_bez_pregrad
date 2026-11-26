package ru.obucheniesluzheniyu.app.recognition

import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import org.junit.AfterClass
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.Parameterized
import ru.obucheniesluzheniyu.app.phrases.loadSignEntries

@RunWith(Parameterized::class)
class SignFlowGestureTest(
    private val word: String,
    private val slug: String,
    private val take: Int,
) {
    @Test
    fun predictsExactlyTheExpectedWordAndLesson() = runBlocking(Dispatchers.IO) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val testAssets = instrumentation.context.assets
        assumeTrue("Supply recorded gestures with -PsignflowControlDir", testAssets.list("mvit-gestures").orEmpty().isNotEmpty())
        val context = instrumentation.targetContext
        val tensor = testAssets.controlTensor("mvit-gestures/$slug/$take.f32")
        val model = recognizer ?: SlovoModel.open(context, loadSignEntries(context.assets)).also { recognizer = it }

        val actual = model.recognizeTensor(tensor)

        assertEquals(word, actual.text)
        assertEquals("https://signflow.ru/$slug", actual.signflowUrl)
    }

    companion object {
        private var recognizer: SlovoModel? = null

        @JvmStatic
        @AfterClass
        fun closeModel() {
            recognizer?.close()
            recognizer = null
        }

        @JvmStatic
        @Parameterized.Parameters(name = "{0}, запись {2}")
        fun cases(): List<Array<Any>> = listOf(
            "Привет" to "privet",
            "Хорошо" to "khorosho",
            "Помочь" to "pomoch",
            "Ждать" to "zhdat",
            "Когда" to "kogda",
            "Завтра" to "zavtra",
            "Утро" to "utro",
            "Вечер" to "vecher",
            "Деньги" to "dengi",
            "Купить" to "kupit",
            "Магазин" to "magazin",
            "Улица" to "ulitsa",
            "Идти" to "idti",
            "Семья" to "semya",
            "Школа" to "shkola",
            "Читать" to "chitat",
            "Писать" to "pisat",
            "Чистый" to "chistyy",
            "Грязный" to "gryaznyy",
            "Собака" to "sobaka",
            "Ключ" to "klyuch",
            "Кружка" to "kruzhka",
            "Вилка" to "vilka",
            "Бутылка" to "butylka",
            "Соль" to "sol",
        ).flatMap { (word, slug) ->
            (1..2).map { take -> arrayOf<Any>(word, slug, take) }
        }
    }
}

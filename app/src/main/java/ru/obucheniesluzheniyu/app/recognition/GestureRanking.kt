package ru.obucheniesluzheniyu.app.recognition

import ru.obucheniesluzheniyu.app.phrases.SignEntry

internal fun bestCatalogGesture(scores: FloatArray, entries: List<SignEntry>): SignEntry =
    entries.maxBy { scores[it.classIndex] }

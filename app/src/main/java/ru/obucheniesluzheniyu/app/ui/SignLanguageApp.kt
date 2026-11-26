package ru.obucheniesluzheniyu.app.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.PrimaryTabRow
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import ru.obucheniesluzheniyu.app.R

private enum class Mode(val tabLabel: Int) {
    Recognition(R.string.recognition_tab),
    Phrases(R.string.phrases_tab),
}

@Composable
fun SignLanguageApp() {
    var mode by rememberSaveable { mutableStateOf(Mode.Recognition) }
    val colorScheme = if (isSystemInDarkTheme()) darkColorScheme() else lightColorScheme()

    MaterialTheme(colorScheme = colorScheme) {
        Scaffold { innerPadding ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(innerPadding)
                    .verticalScroll(rememberScrollState()),
            ) {
                Text(
                    text = stringResource(R.string.app_name),
                    modifier = Modifier.padding(horizontal = 24.dp, vertical = 16.dp),
                    style = MaterialTheme.typography.headlineMedium,
                )
                PrimaryTabRow(selectedTabIndex = mode.ordinal) {
                    Mode.entries.forEach { tab ->
                        Tab(
                            selected = mode == tab,
                            onClick = { mode = tab },
                            text = { Text(stringResource(tab.tabLabel)) },
                        )
                    }
                }
                if (mode == Mode.Recognition) {
                    RecognitionScreen()
                } else {
                    PhrasesScreen()
                }
            }
        }
    }
}

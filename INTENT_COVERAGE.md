# Intent Coverage

Stand: 2026-03-04

| Intent | Beispiel-Input | Primärer Erkennungsweg | Testabdeckung |
|---|---|---|---|
| `smalltalk_begruessung` | `Hallo` | Exact/Fuzzy + Modell | `test_get_returns_response` |
| `smalltalk_stimmung` | `Wie geht es dir?` | Exact/Fuzzy + Modell | `test_get_stimmung_question_returns_stimmung_answer` |
| `mental_checkin_start` | `check in` | Special Rule + Keyword/Fuzzy | `test_plain_check_in_maps_to_mental_checkin` |
| `mental_stress_support` | `Ich bin gestresst` | Keyword + Modell | `test_follow_up_detail_request_after_mental_prompt` (Setup) |
| `mental_anxiety_support` | `Ich habe Panik` | Keyword + Modell | Indirekt über Follow-up-Rotation |
| `mental_sleep_support` | `Ich kann nicht schlafen` | Keyword + Modell | Keine dedizierte Einzelprüfung |
| `mental_focus_support` | `Ich bin total abgelenkt` | Keyword + Fuzzy | `test_flexible_focus_intent_detection` |
| `mental_energy_support` | `Ich bin antriebslos` | Keyword + Fuzzy | `test_keyword_energy_intent_detection` |
| `mental_overthinking_support` | `Gedankenkarussell` | Keyword + Fuzzy | Keine dedizierte Einzelprüfung |
| `mental_breathing_exercise` | `Noch eine Atemuebung` | Exact/Fuzzy + Modell | `test_explicit_breathing_prompt_not_overridden_by_follow_up_marker` |
| `mental_grounding` | `Grounding Uebung` | Exact/Fuzzy + Modell | Indirekt über Follow-up/Buttons |
| `mental_body_scan` | `Body Scan bitte` | Keyword + Fuzzy | `test_body_scan_follow_up_steps` |
| `mental_crisis_support` | `Ich will mir was antun` | Hard Keyword Safety Rule | Keine dedizierte Einzelprüfung |
| `feedback_positiv` | `Danke` | Exact/Fuzzy + Modell | Indirekt in Dialogfluss |
| `feedback_negativ` | `Das hilft nicht` | Exact/Fuzzy + Modell | Indirekt in Dialogfluss |
| `thema_auswahl` | `Welche Themen kannst du?` | Exact/Fuzzy + Modell | Keine dedizierte Einzelprüfung |
| `Fallback` | `Banane Fahrrad Wolke` | Kein Intent getroffen | `test_get_unknown_message_returns_default` |

## Hinweise

- Mehrdeutige Inputs nutzen eine Priorität: `Exact` -> `Keyword` -> `Fuzzy` -> `Modell`.
- Follow-up (`weiter`, `genauer`, `noch`) nutzt Kontext und den letzten Intent.
- Sicherheitskritische Krise-Begriffe werden vor dem Modell abgefangen.

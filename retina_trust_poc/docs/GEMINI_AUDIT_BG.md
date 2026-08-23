# Проверка на предложението от Gemini

## Резюме

Полезната част от предложението е правилна: следващата съществена стъпка е външна оценка с реални quality labels. Първоначалната препоръка обаче не е изпълнима без корекции, защото Gemini не е имал достъп до repository-то и е работил по описан snapshot, а не по реалната v0.3 структура.

## Коригирани твърдения

| Твърдение в предложението | Проверка | Решение в RetinaTrust v0.4 |
|---|---|---|
| Messidor-2 има официални DR и gradability labels | Невярно. Официалният пакет няма DR ground truth. | Messidor-2 не е primary dataset. |
| DeepDRiD е CC BY-NC-SA | Невярно за официалния GitHub repository, който декларира CC BY-SA 4.0. | Използва се официалният repository и неговият LICENSE. |
| Външният набор доказва demographic bias | Методологично прекалено силно. | Докладва се общ cross-dataset shift, който смесва няколко фактора. |
| NPV е достатъчна за quality gate | Непълно. | Добавени са sensitivity, specificity, PPV, NPV, F1, balanced accuracy, under- и over-rejection. |
| Нужна е петкласова confusion matrix | Неприложимо към бинарния модел. | Добавен е binary error анализ, стратифициран по grade 0–4. |
| Случайни 20% могат да се ползват за recalibration | Риск от patient leakage и загуба на external-test статута. | Първичният run е zero-shot; adaptation може само отделно и с patient split. |
| N=1748 решава статистическата нестабилност | Невярно. Двете очи и гледните точки не са независими. | Интервалите са patient-clustered bootstrap. |
| Официалният IDRiD test reuse е „information leak“ и прави резултата невалиден | Прекалено силна формулировка. | Остава честното описание: fixed official split, reused after protocol correction; не е използван за fitting или threshold selection. |
| Специфичността вероятно ще падне под 20% | Неподкрепена спекулация. | Не се публикуват външни числа преди реален run. |

## Проверени първични източници

- [DeepDRiD official repository](https://github.com/deepdrdoc/DeepDRiD)
- [DeepDRiD challenge page](https://biomedicalimaging.org/2020/wp-content/uploads/static-html-to-wp/data/dff0d41695bbae509355435cd32ecf5d/index-29.htm)
- [DeepDRiD paper](https://doi.org/10.1016/j.patter.2022.100512)
- [Messidor-2 official distribution page](https://www.adcis.net/en/third-party/messidor2/)
- [DDR official repository](https://github.com/nkicsl/DDR-dataset)

## Какво запазваме от работата на Gemini

- приоритетът върху независим набор;
- необходимостта от quality ground truth;
- zero-shot резултат преди recalibration;
- отделно разглеждане на diagnostic и quality-gate performance;
- предварително фиксиран протокол.

Тези идеи са реализирани в действителната структура `retina_poc/`, `scripts/`, `tests/` и `artifacts/v0.4/`, без несъществуващите `src/` пътища от предложението.

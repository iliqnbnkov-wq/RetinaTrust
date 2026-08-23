# Предварително фиксиран протокол за външна валидация

## Статус

Този документ описва протокола на RetinaTrust v0.4 **преди да са изчислени външни резултати**. Няма външни AUROC, sensitivity, specificity или quality-gate числа в този release. Това е умишлено: кодът, моделът, праговете и статистическият план се заключват преди преглед на резултатите.

Външната оценка е изследователска. Тя не превръща RetinaTrust в диагностична система, медицинско изделие или клинично валидиран продукт.

## Защо DeepDRiD е основният набор

Основният външен набор е regular-fundus частта на [DeepDRiD](https://github.com/deepdrdoc/DeepDRiD). Тя е подходяща, защото съдържа едновременно:

- patient ID и по две гледни точки за всяко око;
- DR степен за ляво и дясно око и patient-level DR степен;
- бинарен `Overall quality` етикет;
- отделни оценки за clarity, field definition и artifact.

Официалната публикация описва общо 2 000 regular-fundus изображения от 500 пациенти: [Liu et al., Patterns 2022](https://doi.org/10.1016/j.patter.2022.100512). Първичният protocol run използва публикуваните CSV training и validation части. Online challenge evaluation частта не се смесва автоматично, защото е публикувана с различен формат на етикетите.

Messidor-2 не е избран за основен набор. [Официалната страница](https://www.adcis.net/en/third-party/messidor2/) изрично посочва, че самият пакет не съдържа DR ground truth; отделните third-party annotations не са част от официалната услуга. Следователно той не дава единен, официален DR + gradability target за настоящия протокол.

## Заключени компоненти

Преди изпълнение `scripts/evaluate_external.py` проверява `artifacts/v0.4/external_protocol_lock.json`. Заключени са:

- моделът `artifacts/v0.3/retina_baseline.joblib`;
- canonical preprocessing и feature extraction;
- quality scoring и workflow правилата;
- external-validation parser, metrics и runner;
- classification threshold 0.50;
- confidence floor 0.85;
- 2 000 patient-clustered bootstrap повторения и фиксиран seed.

При hash mismatch изпълнението спира. Няма CLI параметър за смяна на праговете или bootstrap плана.

## Популация и включване

Включват се всички редове от официалните:

1. `regular-fundus-training.csv`;
2. `regular-fundus-validation.csv`.

CSV схемата се проверява строго. Липсващо изображение, дублиран `image_id`, невалидна DR степен или несъответствие между `_l/_r` в името и eye-grade колоната прекратява run-а. Няма casewise изключване след виждане на моделния резултат.

`Overall quality = 1` се интерпретира като достатъчно качество, а `0` като недостатъчно качество съгласно DeepDRiD label definition. DR степени 0–4 се преобразуват предварително към:

```text
referable DR = eye DR grade >= 2
```

Ако се срещне grade 5, изображението остава в quality анализа, но се изключва от binary DR анализа като ungradable. Изключването се отчита в denominators.

## Първични анализи

### 1. DR на ниво изображение

Първичната единица е отделното изображение, защото замразеният v0.3 модел приема една снимка. Изчисляват се:

- AUROC;
- sensitivity и specificity при threshold 0.50;
- PPV и NPV;
- F1, Macro-F1 и balanced accuracy;
- Brier score, negative log-likelihood и ECE;
- 2×2 confusion matrix.

Няма универсален „клинично приемлив“ AUROC праг. Числата се докладват с ограниченията им, без post-hoc pass/fail етикет.

### 2. Quality gate спрямо `Overall quality`

Положителното събитие е **официално недостатъчно качество**. Докладват се два предварително зададени режима:

- intervention: RetinaTrust `review` или `fail`;
- recapture: RetinaTrust `fail`.

За всеки режим се дават sensitivity към ungradable, specificity към gradable, PPV, NPV, F1, balanced accuracy и confusion matrix. Допълнително:

- under-rejection = ungradable изображение, пропуснато като `pass`;
- over-rejection = gradable изображение, изпратено към intervention/recapture.

NPV самостоятелно не е достатъчна оценка на quality gate.

## Зависимост между изображенията

Един пациент има няколко изображения. Затова 95% интервалите се изчисляват с nonparametric bootstrap, при който се семплират **пациенти с възстановяване**, а всички техни снимки се запазват като cluster. Наивен image-level bootstrap би подценил зависимостта между гледните точки.

## Selective и вторични анализи

Винаги първо се показва резултатът върху всички допустими изображения. След това се показват:

- quality-`pass` subset заедно с coverage;
- quality-`pass` + confidence ≥ 0.85 subset заедно с coverage;
- binary errors, стратифицирани по оригинална DR степен 0–4.

Няма петкласова confusion matrix, защото RetinaTrust е бинарен модел. Grade-stratified таблицата показва поведението на бинарното решение във всяка оригинална степен.

Вторично и ясно означено се изчисляват:

- eye-level probability = максимумът от наличните гледни точки за окото;
- patient-level probability = максимумът от изображенията на пациента.

Тези max правила са предварително зададени, но могат да увеличат false positives и не се представят като установен клиничен стандарт.

## Забранени действия в първичния run

- обучение, recalibration или избор на праг върху DeepDRiD;
- избор на subset след виждане на резултатите;
- смяна на quality прагове;
- представяне на cross-dataset shift като изолиран demographic-bias тест;
- представяне на selective метрики без coverage;
- смесване на synthetic perturbation резултатите с реалните quality labels.

Ако по-късно се изследва recalibration, тя трябва да е отделен adaptation experiment с patient-level разделяне и отделен untouched evaluation subset. Тя не заменя zero-shot резултата.

## Възпроизводим run

След изтегляне на официалния DeepDRiD repository:

```bash
python scripts/verify_artifacts.py
python -m unittest discover -s tests -v
python scripts/evaluate_external.py --dataset-root "/path/to/DeepDRiD"
```

По подразбиране резултатите се записват в `outputs/deepdrid_external/`, която не се включва автоматично в release. Преди публикуване трябва да се добавят dataset label hashes, review на denominators, проверка на JSON/CSV и нов evidence manifest.

## Ограничение на извода

Cross-dataset промяната смесва разлики в популация, камера, acquisition workflow и labeling. Затова резултатът измерва обща външна преносимост към този набор; не изолира демографска справедливост, клинична безопасност или причината за евентуален performance drop.

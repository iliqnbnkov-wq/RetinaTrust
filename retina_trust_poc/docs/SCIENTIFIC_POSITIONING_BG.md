# Научно позициониране на RetinaTrust v0.4

Емпиричните резултати остават от проверения v0.3 baseline. v0.4 добавя предварително фиксиран DeepDRiD протокол; външни резултати още няма.

## Краткият честен отговор

RetinaTrust не е първата система, която оценява качеството на фундус изображения, измерва несигурност или отказва автоматично решение. Такава претенция би била невярна.

Академичната стойност е в малък, локален и възпроизводим workflow, който събира:

1. фиксираното официално разделяне на IDRiD;
2. отделни технически фактори за качество;
3. калибриран baseline за referable DR;
4. пет контролирани нарушения с три степени;
5. общ canonical preprocessing за training, clean и degraded изображения;
6. per-image prediction audit и paired bootstrap интервали;
7. отделни решения за повторно заснемане, човешки преглед и предварителен резултат.

Това е интеграционен, експериментален и образователен принос. Не е нова невронна архитектура и не е клинична валидация.

## Какво вече е направено

- EyeQ показва, че DR производителността зависи от качеството на входа: [Fu et al., MICCAI 2019](https://doi.org/10.1007/978-3-030-32239-7_6).
- Факторна оценка чрез артефакти, яснота и поле на видимост е разработена преди RetinaTrust: [Shen et al., Medical Image Analysis 2020](https://doi.org/10.1016/j.media.2020.101654).
- Несигурност и насочване на трудните случаи са изследвани при DR: [Leibig et al., Scientific Reports 2017](https://doi.org/10.1038/s41598-017-17876-z) и [DR|GRADUATE, Medical Image Analysis 2020](https://doi.org/10.1016/j.media.2020.101715).
- Контролирани нарушения, външни набори и quality-aware gate са комбинирани в близка работа: [Issayev and Ziro, 2025](https://doi.org/10.32743/UniTech.2025.134.5.20125).
- Повторно заснемане и confidence-guided multi-image fusion са изследвани в [Raghu et al., arXiv:2607.03643, 2026](https://doi.org/10.48550/arXiv.2607.03643). Това е preprint, не доказателство за установена клинична практика.

Правилната формулировка е:

> RetinaTrust реализира проверим workflow, който показва как конкретно нарушение променя агрегатните метрики, отделните решения, увереността и правилото за намеса.

## Какво показва v0.3

- Clean fixed test split: AUROC 0.720 (95% CI 0.619–0.812), sensitivity 0.859, specificity 0.359, Macro-F1 0.608 и ECE 0.087.
- Severe underexposure: AUROC 0.546; paired ΔAUROC −0.174 (95% CI −0.287 до −0.056); prediction flips 77.7% (68.9–85.4%).
- Severe gamma deviation: AUROC 0.602; paired ΔAUROC −0.118 (−0.213 до −0.022); flips 40.8%.
- При severe colour shift, blur и JPEG 95% CI за ΔAUROC пресича нулата. Спадът в discriminative performance не е установен убедително върху n=103, въпреки че 20.4%, 36.9% и 41.7% от thresholded решенията се сменят.
- Quality gate се намесва във всички severe underexposure и gamma случаи, но самият gate не е клинично валидиран.

Prediction flip означава нестабилност спрямо clean изхода, не задължително нова грешка. Затова flip rate и performance metric трябва да се четат заедно.

## Как беше поправен v0.1

Във v0.1 нарушените изображения минаваха през 1024 px resampling, а clean резултатът идваше от директния оригинален път. Това беше реален confound.

Във v0.3:

1. training, clean и degraded входовете използват еднакъв canonical RGB/LANCZOS път;
2. clean predictions се възпроизвеждат с максимална грешка 1.11×10⁻¹⁶;
3. пазят се всички per-image вероятности;
4. разликите имат 2 000-sample paired bootstrap 95% CI;
5. старият модел има отделен resampling-only control.

Resampling-only контролът причинява 6.8% flips (95% CI 2.9–11.7%), но ΔAUROC е −0.006 (−0.030 до +0.018). Значи старият pipeline е внасял нестабилност, но този артефакт не обяснява големия severe-underexposure ефект.

## Ограничение на test split-а

Официалните 103 test изображения вече са били разглеждани чрез v0.1 резултатите. Във v0.3 не са използвани за fitting, calibration, model selection или избор на праг, но не трябва да се наричат „никога невиждан заключен test set“. Честното описание е „фиксиран official test split, повторно използван след корекция на протокола“.

Нужни са външен набор, друг център/камера и клинични gradability labels.

## Какво добавя v0.4 без да надценява доказателствата

v0.4 подготвя zero-shot оценка върху DeepDRiD regular fundus. Наборът има DR степени, `Overall quality` и patient ID, затова позволява отделно да се оценят бинарният DR модел и quality gate-ът с patient-clustered bootstrap интервали.

Протоколът е заключен преди резултатите: моделът, classification threshold 0.50, confidence floor 0.85, quality правилата, кодът и bootstrap планът имат hash записи. Няма fitting, recalibration или post-hoc threshold selection върху DeepDRiD.

Това ще бъде външна cross-dataset оценка, но не изолира demographic bias. Евентуална промяна може едновременно да отразява популация, камера, acquisition workflow и labeling differences. Messidor-2 не се използва като primary dataset, защото официалният му пакет няма DR ground truth.

## Допустими твърдения

- „Това е възпроизводим изследователски PoC за quality-conditioned reliability.“
- „Severe underexposure и gamma имат paired AUROC интервали под нулата в тази извадка.“
- „Показваме case-level instability, а не само една обща accuracy.“
- „Quality gate демонстрира workflow за отказ, но не е клинично валидиран.“
- „Поправихме preprocessing confound-а и измерихме остатъчния му ефект.“

## Недопустими твърдения

- „Системата диагностицира диабетна ретинопатия.“
- „Прагът 0.85 или quality score са медицински валидирани.“
- „RetinaTrust е първата система с quality gate, uncertainty или retake.“
- „77.7% е универсален ефект при всички камери, модели и популации.“
- „Test set-ът е останал напълно заключен и невиждан.“
- „CI под нулата доказва клинична причинност.“

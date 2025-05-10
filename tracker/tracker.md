# Майский чекпоинт (улучшение трекера)

В качестве майского чекпоинта я поставил себе цель улучшить вторую составляющую проекта — трекер BoT-SORT. Основной его проблемой является его скорость работы, а именно скорость алгоритма GMC (Global Motion Compensation) — расчёта того, как между каждой соседней парой кадров сдвигается камера. Существуют разные методы GMC. Метод Sparse Optical Flow, используемый по умолчанию в библиотеке YOLO, занимает около 90% времени трекинга. Я исследовал следующие альтернативные методы GMC:

- модель [RAFT](https://arxiv.org/abs/2003.12039),
- способы ускорить Sparse Optical Flow (в основном путём уменьшения размера изображения перед подачей его в метод).

При помощи модели RAFT у меня получилось побить коробочный BoT-SORT и по скорости, и по качеству. Модифицированный Sparse Optical Flow не доходит до RAFT по качеству, но позволяет добиться ещё большего ускорения: весь пайплайн детекции и трекинга можно ускорить в 2 раза.

## Методология экспериментов

Я сравнивал скорость и качество всей системы вместе (детектор+трекер). Скорость и качество замерялись на валидационной выборке датасета SportsMOT. Скорость замерялась как среднее время на обработку одного кадра в миллисекундах. Качество замерялось по метрике HOTA с помощью библиотеки TrackEval. В качестве детектора для большинства экспериментов использовалась лучшая модель с мартовского чекпоинта — дообученная yolo11l (march-best). Также я замерил лучшие методы на аналогично дообученной yolo11s (march-best-s). Замеры производились в сервисе Kaggle (GPU P100).

## RAFT

RAFT — модель, которая рассчитывает local optical flow между двумя изображениями, т. е. для каждого пикселя выдаёт 2 числа — насколько он сдвинулся по координатам x и y. Для метода BoT-SORT нам нужен только global optical flow, т. е. 2 числа, отображающие движение всего изображения сразу. Поэтому выходы RAFT нужно как-то агрегировать. Я беру медиану всех пикселей по x и по y (я пробовал брать среднее, но это сработало хуже). Также я сделал следующие модификации, которые дали улучшение по скорости без потери качества:

- упростил архитектуру, убрав последний слой, который upsample'ирует итоговые потоки (т. к. мы всё равно в конце их все агрегируем),
- подавал на вход модели изображение, уменьшенное до размера 224х128 пикселей,
- делал всего одну итерацию обновления потока вместо нескольких, рекомендуемых в статье.

Конкретно я использовал предобученную [RAFT-S](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.optical_flow.raft_small.html) с DEFAULT весами из библиотеки torchvision.

## Ускорение Sparse Optical Flow

Я обнаружил, что в нашей задаче Sparse Optical Flow может неплохо работать и на очень сильно уменьшенных изображениях. Это сильно ускоряет алгоритм, при этом потери качества не так велики, как я ожидал. Даже уменьшение изображения в 20 раз до размера 64x36 пикселей может иметь смысл, если требуется максимальная скорость трекинга.

Я также пробовал уменьшить количество "углов", по которым рассчитывается движение камеры в алгоритме, но это оказалось не так эффективно, как простое уменьшение входного изображения.

## Основные результаты

Основные результаты приведены в табличке ниже. Здесь везде используется детектор march-best (текущая лучшая дообученная yolo11l) и трекер BoT-SORT. После знака ± указано стандартное отклонение по всем кадрам в валидационной выборке. Напомню, что время на кадр — это время, потраченное суммарно на детекцию и на трекинг.

| детектор | метод GMC | время на кадр, мс | HOTA |
| :------- | :------- | :----------------: | :----: |
| yolo11l, march-best | Sparse Optical Flow из коробки | 48\.3 ± 3\.7 | 67\.788 |
| yolo11l, march-best | RAFT, лучший вариант | 32\.3 ± 2\.8 | **68\.350** |
| yolo11l, march-best | Sparse Optical Flow, 8x downscale | 27\.7 ± 2\.2 | 67\.845 |
| yolo11l, march-best | Sparse Optical Flow, 10x downscale | 26\.2 ± 2\.5 | 67\.100 |
| yolo11l, march-best | Sparse Optical Flow, 16x downscale | 23\.7 ± 1\.7 | 66\.321 |
| yolo11l, march-best | Sparse Optical Flow, 20x downscale | **23\.1 ± 1\.8** | 66\.061 |

Результаты этих же методов с уменьшенным детектором на основе yolo11s:

| детектор | метод GMC | время на кадр, мс | HOTA |
| :------- | :------- | :----------------: | :----: |
| yolo11s, march-best-s | Sparse Optical Flow из коробки | 38\.0 ± 3\.4 | 66\.146 |
| yolo11s, march-best-s | RAFT, лучший вариант | 24\.6 ± 2\.8 | **66\.510** |
| yolo11s, march-best-s | Sparse Optical Flow, 8x downscale | 20\.1 ± 1\.8 | 66\.199 |
| yolo11s, march-best-s | Sparse Optical Flow, 10x downscale | 18\.1 ± 2\.3 | 65\.576 |
| yolo11s, march-best-s | Sparse Optical Flow, 16x downscale | 17\.2 ± 1\.9 | 64\.418 |
| yolo11s, march-best-s | Sparse Optical Flow, 20x downscale | **15\.3 ± 1\.0** | 64\.480 |

## Все эксперименты

В табличке ниже в менее формальном виде приведены все эксперименты, которые я поставил для этого чекпоинта.

| детектор | трекер | время на кадр, мс | HOTA |
| :-------- | :------ | :-----------------: | :----: |
| yolo11l, march-best | botsort, sparseOptFlow | 48\.3 ± 3\.7 | 67\.788 |
| yolo11l, march-best | botsort, sparseOptFlow, new thresholds | 47\.3 ± 3\.3 | 67\.035 |
| yolo11l, march-best | botsort, sparseOptFlow, 144p \(5x downscale\) | 41\.2 ± 4\.2 | 68\.021 |
| yolo11l, march-best | botsort, sparseOptFlow, 90p \(8x downscale\) | 27\.7 ± 2\.2 | 67\.845 |
| yolo11l, march-best | botsort, sparseOptFlow, 72p \(10x downscale\) | 26\.2 ± 2\.5 | 67\.100 |
| yolo11l, march-best | botsort, sparseOptFlow, 60p \(12x downscale\) | 29\.3 ± 3\.0 | 66\.894 |
| yolo11l, march-best | botsort, sparseOptFlow, 45p \(16x downscale\) | 23\.7 ± 1\.7 | 66\.321 |
| yolo11l, march-best | botsort, sparseOptFlow, 36p \(20x downscale\) | 23\.1 ± 1\.8 | 66\.061 |
| yolo11l, march-best | botsort, sparseOptFlow, 500 corners | 38\.4 ± 2\.4 | 67\.088 |
| yolo11l, march-best | botsort, raft-s, 128p, 1 update, medians | 33\.7 ± 3\.5 | 68\.320 |
| yolo11l, march-best | botsort, raft-s, 128p, 1 update, no upsample, medians | 32\.3 ± 2\.8 | 68\.350 |
| yolo11l, march-best | botsort, raft-s, 144p, 1 update, medians | 36\.7 ± 3\.8 | 68\.046 |
| yolo11l, march-best | botsort, raft-s, 144p, 1 update, no upsample, medians | 35\.2 ± 3\.1 | 68\.230 |
| yolo11l, march-best | botsort, raft-s, 144p, 3 updates, medians | 37\.4 ± 3\.0 | 67\.740 |
| yolo11l, march-best | botsort, raft-s, 144p, 5 updates, medians | 41\.4 ± 3\.6 | 67\.780 |
| yolo11l, march-best | botsort, raft-s, 288p, 1 update, medians | 36\.9 ± 3\.0 | 67\.738 |
| yolo11l, march-best | botsort, raft-s, 720p, 1 update, medians | 129\.7 ± 11\.4 | 67\.665 |
| yolo11l, march-best | botsort, raft-s, 144p, 1 update, means | 32\.8 ± 2\.8 | 67\.691 |
| yolo11l, march-best | botsort, raft-s, 128p upper middle crop, 1 update, medians | 30\.8 ± 3\.0 | 65\.914 |
| yolo11l, march-best | botsort, raft-s, 128p square crop, 1 update, medians | 39\.6 ± 3\.4 | 67\.735 |
| yolo11l, march-best | botsort, no gmc | 22\.4 ± 2\.7 | 64\.161 |
| yolo11l, march-best | bytetrack, old thresholds | 23\.2 ± 1\.9 | 58\.519 |
| yolo11l, march-best | bytetrack, new thresholds | 21\.8 ± 2\.2 | 57\.299 |
| yolo11l, march-best | none, detection only | 19\.8 ± 1\.1 | n/a |
| yolo11m, stock | none, detection only | 16\.9 ± 0\.7 | n/a |
| yolo11s, march-best-s | botsort, sparseOptFlow | 38\.0 ± 3\.4 | 66\.146 |
| yolo11s, march-best-s | botsort, sparseOptFlow, 90p \(8x downscale\) | 20\.1 ± 1\.8 | 66\.199 |
| yolo11s, march-best-s | botsort, sparseOptFlow, 72p \(10x downscale\) | 18\.1 ± 2\.3 | 65\.576 |
| yolo11s, march-best-s | botsort, sparseOptFlow, 45p \(16x downscale\) | 17\.2 ± 1\.9 | 64\.418 |
| yolo11s, march-best-s | botsort, sparseOptFlow, 36p \(20x downscale\) | 15\.3 ± 1\.0 | 64\.480 |
| yolo11s, march-best-s | botsort, raft-s, 128p, 1 update, no upsample, medians | 24\.6 ± 2\.8 | 66\.510 |
| yolo11s, march-best-s | botsort, raft-s, 144p, 1 update, medians | 25\.4 ± 2\.6 | 66\.246 |
| yolo11s, march-best-s | botsort, no gmc | 14\.1 ± 2\.1 | 62\.066 |
| yolo11s, baseline | none, detection only | 12\.6 ± 0\.8 | n/a |

Я также пробовал запустить другие методы GMC, доступные в библиотеке YOLO (ORB, SIFT и ECC), однако стало ясно, что они не подходят для real-time применения, т. к. требуют более 200 миллисекунд на кадр. Поэтому я не делал полноценные их замеры.

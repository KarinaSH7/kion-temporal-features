"""Create the Russian report from recorded experiment outputs, never invented scores."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'tlab-mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,KeepTogether

from .data import ROOT,dump

LABELS={'manual':'Ручные признаки','auto_all':'Все автопризнаки','auto_selected':'Отобранные автопризнаки',
        'no_indirect':'Без косвенных признаков','popularity':'Популярность за 7 дней'}


def table_md(rows,headers):
    def row(x): return '| '+' | '.join(map(str,x))+' |'
    return '\n'.join([row(headers),row(['---']*len(headers))]+[row(r) for r in rows])


def main():
    out=ROOT/'results/main'; reports=ROOT/'reports'; reports.mkdir(exist_ok=True)
    metrics=pd.read_csv(out/'test_metrics.csv')
    lock=json.loads((out/'locked_config.json').read_text())
    manifest=json.loads((ROOT/'results/data_manifest_10000.json').read_text())
    ci=json.loads((out/'paired_bootstrap.json').read_text())
    stats={name:json.loads((out/f'{name}_statistics.json').read_text()) for name in manifest['cutoffs']}
    trials=pd.read_csv(out/'tuning_trials.csv')
    search=pd.read_csv(out/'search_trials.csv')
    runtime=pd.read_csv(out/'runtime.csv')
    oracle=json.loads((out/'candidate_oracle.json').read_text())
    scaling=pd.read_csv(out/'scaling.csv')
    selected=lock['chosen']['auto_selected']['columns']
    manual=lock['chosen']['manual']['columns']
    added=[c for c in selected if c not in manual]
    importance=pd.read_csv(out/'auto_selected_importance.csv')
    assert np.isfinite(importance.importance).all(), 'Feature importance export is missing'
    primary=metrics[(metrics.seed==42)&(metrics.arm=='auto_selected')].iloc[0]
    base=metrics[(metrics.seed==42)&(metrics.arm=='manual')].iloc[0]
    all_auto=metrics[(metrics.seed==42)&(metrics.arm=='auto_all')].iloc[0]
    ablated=metrics[(metrics.seed==42)&(metrics.arm=='no_indirect')].iloc[0]
    popular=metrics[(metrics.seed==42)&(metrics.arm=='popularity')].iloc[0]
    delta=ci['42']['delta']; relative=100*delta/base.ndcg10
    deltas=[ci[str(seed)]['delta'] for seed in [42,17,2026]]
    robustness=('Знак эффекта меняется между seed: улучшение неустойчиво к случайности обучения.'
                if min(deltas)<0<max(deltas) else
                'Знак эффекта одинаков во всех трёх seed, но это не заменяет проверку на новых временных срезах.')
    significant=ci['42']['ci95_low']>0
    verdict=('На основном seed автоматический отбор улучшил NDCG@10; парный 95%-й интервал не пересекает ноль.' if significant else
             'На основном seed статистически убедительное превосходство автоматического отбора не установлено: парный 95%-й интервал не лежит целиком выше нуля.')
    rows=[]
    for arm in ['popularity','no_indirect','manual','auto_all','auto_selected']:
        r=metrics[(metrics.arm==arm)&(metrics.seed==42)].iloc[0]
        rows.append([LABELS[arm],int(r.n_features),f'{r.ndcg10:.5f}',f'{r.recall10:.5f}',f'{r.auc:.5f}',
                     '-' if pd.isna(r.get('logloss')) else f'{r.logloss:.5f}'])
    results_table=table_md(rows,['Метод','F','NDCG@10','Recall@10','ROC-AUC','LogLoss'])
    seed_rows=[]
    for seed in [42,17,2026]:
        a=metrics[(metrics.arm=='manual')&(metrics.seed==seed)].iloc[0]
        b=metrics[(metrics.arm=='auto_selected')&(metrics.seed==seed)].iloc[0]
        c=ci[str(seed)]
        seed_rows.append([seed,f'{a.ndcg10:.5f}',f'{b.ndcg10:.5f}',f'{c["delta"]:+.5f}',f'[{c["ci95_low"]:+.5f}; {c["ci95_high"]:+.5f}]'])
    seed_table=table_md(seed_rows,['Seed','Manual','Selected','Разность','95% CI разности'])
    split_rows=[]
    for name,s in stats.items():
        split_rows.append([name,s['cutoff'][:10],s['end'][:10],s['queries'],s['candidate_rows'],s['positive_queries'],s['candidate_positives']])
    split_table=table_md(split_rows,['Срез','Начало','Конец, не включён','Запросы','Пары','Положит. запросы','Положит. пары'])
    configs=[]
    for arm,s in lock['chosen'].items():
        configs.append([LABELS[arm],len(s['columns']),s['depth'],s['iterations'],f'{s["validation"]["ndcg10"]:.5f}'])
    config_table=table_md(configs,['Метод','F','Глубина','Деревья','Tune NDCG@10'])
    costs=[]
    for arm in lock['chosen']:
        r=runtime[(runtime.arm==arm)&(runtime.seed==42)].iloc[0]
        cost=trials[trials.arm==arm].groupby('depth').full_fit_seconds.first().sum()
        costs.append([LABELS[arm],f'{cost:.1f}',f'{r.inference_seconds:.3f}',f'{r.model_bytes/1048576:.2f}'])
    cost_table=table_md(costs,['Метод','2 обучающих запуска, с','Inference всего теста, с','Модель, MiB'])
    selected_importance=importance[importance.feature.isin(added)].head(10)
    top_auto='; '.join(f'{r.feature} ({r.importance:.2f}%)' for r in selected_importance.itertuples())
    top_all='; '.join(f'{r.feature} ({r.importance:.2f}%)' for r in importance.head(8).itertuples())
    test=stats['test']
    dates='; '.join(f'{k}: {v}' for k,v in manifest['cutoffs'].items())
    mean=metrics[metrics.arm.isin(['manual','auto_selected'])].groupby('arm').ndcg10.agg(['mean','std'])
    mean_text=(f"Среднее NDCG@10 по трём seed: manual {mean.loc['manual','mean']:.5f} "
               f"(SD {mean.loc['manual','std']:.5f}), selected {mean.loc['auto_selected','mean']:.5f} "
               f"(SD {mean.loc['auto_selected','std']:.5f}). Это разброс обучения на одной когорте, не три независимых датасета.")
    ndcg_all=f"NDCG@10 по всем запросам, включая пользователей без будущего просмотра: manual {base.ndcg10_all:.5f}, selected {primary.ndcg10_all:.5f}."
    notes=(f"На тесте {test['queries']} запросов, из них {test['positive_queries']} с будущими положительными объектами. "
           f"Recall retrieval@100 = {primary.candidate_recall100:.4f} (macro по таким пользователям); "
           f"доля целевых объектов в известном каталоге = {test['known_catalog_target_coverage']:.4f} (micro). "
           "Эти величины имеют разные знаменатели и не должны напрямую сравниваться как одна метрика.")

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(8.5,3.8),layout='constrained')
    seed42=metrics[metrics.seed==42].set_index('arm').reindex(['popularity','no_indirect','manual','auto_all','auto_selected'])
    yy=np.arange(len(seed42)); vals=seed42.ndcg10.to_numpy()
    ax.barh(yy,vals,color=['#c4cdd9','#9ca9bc','#294768','#63a89c','#147d6f'])
    ax.set_yticks(yy,[LABELS[a] for a in seed42.index]);ax.invert_yaxis()
    ax.set_xlabel('NDCG@10 • тест • seed 42');ax.set_xlim(0,max(vals)*1.22)
    for i,v in enumerate(vals): ax.text(v+.001,i,f'{v:.4f}',va='center')
    fig.savefig(reports/'quality.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,3.0),layout='constrained')
    ax.plot(scaling.n_pairs,scaling.candidate_seconds,'o-',label='Поиск кандидатов')
    ax.plot(scaling.n_pairs,scaling.feature_seconds,'o-',label='193 признака')
    ax.set_xlabel('Число пар');ax.set_ylabel('Секунды');ax.legend(frameon=False);ax.grid(alpha=.2)
    fig.savefig(reports/'scaling.png',dpi=160);plt.close(fig)
    shown=importance.head(10).iloc[::-1]
    fig,ax=plt.subplots(figsize=(8.5,3.8),layout='constrained')
    ax.barh(shown.feature,shown.importance,color=['#147d6f' if c in added else '#294768' for c in shown.feature])
    ax.set_xlabel('PredictionValuesChange, % • train • selected model')
    ax.tick_params(axis='y',labelsize=8)
    fig.savefig(reports/'importance.png',dpi=160);plt.close(fig)

    readme=f'''# T Lab · Автоматический поиск временных признаков для рекомендаций

**Воспроизводимый эксперимент на KION · Python / CatBoost · 10 000 пользователей**

## Краткий результат

{verdict} NDCG@10: **{base.ndcg10:.5f} → {primary.ndcg10:.5f}**, изменение
**{delta:+.5f}** ({relative:+.2f}% относительно baseline), парный 95% CI
**[{ci['42']['ci95_low']:+.5f}; {ci['42']['ci95_high']:+.5f}]**.

{mean_text}
**Устойчивость:** {robustness}

В ручном baseline 64 признака. Грамматика строит 193 признака; отбор сохранил
{len(selected)} (64 ручных + {len(added)} дополнительных). Все числа ниже получены реальными
запусками; результаты поиска, модели, конфигурации и логи сохранены. Это локальная
исследовательская работа, а не результат официального leaderboard KION.

![Качество на тесте](reports/quality.png)

## 1. Постановка задачи и данные

Для пользователя u в момент t требуется ранжировать ранее не наблюдавшиеся фильмы.
Положительный объект — зарегистрированное взаимодействие в интервале [t, t+7 дней).
Отсутствие записи в этом интервале даёт метку 0. Это implicit-feedback задача:
неизвестно, был ли объект показан. Нулевую метку нельзя интерпретировать как
доказанный отказ, а выход модели — как CTR.

Использован официальный [KION](https://github.com/irsafilo/KION_DATASET), commit
`{manifest['source_revision']}`. Полный архив `data_en.zip` содержит
{manifest['population_rows']:,} строк, {manifest['population_users']:,} пользователей и
{manifest['population_items']:,} объектов. Реальные даты файла: 2021-03-13—2021-08-22;
они проверены в данных, а не взяты из описания репозитория. Русские метаданные
объектов загружены из `data_original/items.csv` той же версии.

**Важная проверка источника.** Отдельный `interactions.csv.zip` в этом commit содержит
лишь 1 594 787 строк, последняя дата — `20`; распакованный размер ровно 50 MiB.
Архив проходит CRC, но содержимое обрезано. Он НЕ использован в эксперименте.
Загрузчик проверяет SHA-256, CRC, ожидаемые 5 476 251 строк, формат дат, отсутствие
пропусков и уникальность пары user–item. Это существенно для сохранения истории.

Выборка: случайные 10 000 пользователей (seed=42) среди известных до первого
обучающего среза. Для каждого сохранены ВСЕ строки исходного набора, включая
будущие строки, которые далее изолируются временными фильтрами. Получено
{manifest['rows']:,} взаимодействий с {manifest['items']:,} объектами. Проверено
равенство числа строк по каждому пользователю в исходнике и выборке. Когорта
заранее фиксирована и смещена в сторону пользователей, уже известных к 12 июля;
выводы не распространяются автоматически на новых пользователей.

KION хранит **одну агрегированную строку на пару** с `last_watch_dt`, а не полный
журнал всех сеансов. Здесь эта строка трактуется как псевдособытие в указанную дату.
`total_dur` и `watched_pct` не используются: они агрегированы и не имеют надёжной
исторической доступности. Сохраняется вся опубликованная история, но восстановить
все реальные повторные просмотры из неё нельзя. Будущая повторная активность могла
перенести `last_watch_dt` и скрыть более ранний просмотр. Поэтому гарантируется
point-in-time корректность относительно **опубликованных псевдособытий**, а не
полное восстановление реальной онлайн-истории. Это ограничение выбранного датасета.

Используются статические характеристики объекта (жанр, страна, режиссёр, тип,
год выпуска, возрастной рейтинг, for_kids). Предполагается, что они доступны при
появлении объекта; исторических версий метаданных нет. Динамические демографические
характеристики пользователей и текстовые описания не используются.

## 2. Временное разбиение и защита от утечек

{split_table}

История каждого среза: строго `timestamp < cutoff`. Окно длины w: `[t-w,t)`.
Цели: `[t,t+7d)`. Границы полуоткрытые; одинаковые timestamps целиком остаются
по одну сторону границы. Все обучающие метки известны до начала validation-search.
Модель после подбора не переобучается на валидации: сравнение использует один train.
При тестовом прогнозе история обновляется событиями до 16 августа, в том числе
прошлыми валидационными событиями. Это допустимая информация в момент прогноза.

Пользователи запроса определяются только по прошлому: минимум одно псевдособытие
за 28 дней. На каждом из трёх train-срезов берутся до 2 000 таких пользователей
случайно, с фиксированным seed; validation/test используют всех подходящих.
Это ограничение числа обучающих запросов, а не усечение их истории. Пользователи
без будущих positives остаются в обучении и в таблице кандидатов.

Запрещены и проверяются: будущие события в агрегатах и графе, будущее появление
объектов в каталоге, добавление правильного ответа в кандидаты, random split строк,
настройка по тесту. `src/features.py` разделяет retrieval/features и attach_labels;
будущие labels не передаются в функции построения кандидатов или признаков.

10 модульных тестов проверяют строгую границу времени, левую границу окна,
инвариантность features/retrieval при добавлении будущих событий, отсутствие
просмотренных/будущих объектов в retrieval, сохранение полной истории, формулы
NDCG/Recall с пропущенным positive, tie-breaking, bootstrap и конец горизонта.
Дополнительно `src.audit` сверяет 100 случайных строк каждого среза с исходными
событиями прямым пересчётом 12 условий на строку; для основного сравнения
независимо пересчитывает итоговые NDCG и Recall из сохранённых предсказаний.

## 3. Фиксированный поиск кандидатов

Каталог каждого среза — объекты, встречавшиеся до cutoff в истории выбранной
когорты. Просмотренные пары исключаются. Четыре канала предлагают до 100 объектов:

1. Популярность за 7 дней.
2. Популярность за всю доступную историю.
3. Item-item cosine по бинарной user–item матрице. Similarity умножается на
   `co_count / (co_count + 10)`; у каждого объекта оставляются 50 соседей.
4. Сходство пользовательской истории со всеми жанрами кандидата, умноженное
   на `log(1 + item_popularity)`.

Каналы объединяются reciprocal-rank fusion: `sum(1/(60+rank))`.
В финале остаются 100 объектов, ties разрешаются по item_id. Никаких случайных
отрицательных объектов или вставки будущих положительных объектов нет.
Все методы используют **одни и те же сохранённые строки**, их hash записан для
каждого среза. Это сравнение reranker при фиксированном retrieval, не сравнение
разных end-to-end retrieval-алгоритмов.

{notes}

Диагностический потолок NDCG@10 при идеальном порядке уже извлечённых кандидатов:
**{oracle['ndcg10']:.5f}**. Для него labels используются только после завершения
retrieval, как теоретическая граница, не как обучаемый метод или подбор модели.

## 4. Сильный ручной baseline

CatBoostClassifier с LogLoss обучается на всех 600 000 train-парах. Отрицательные
строки дополнительно не сэмплируются, class weights не используются. 64 признака:

- пользователь: числа взаимодействий за 7/30 дней и всё время, уникальные объекты,
  активные дни, разнообразие жанров, давность последнего события;
- объект: частоты и доли за те же окна, давность, статические категории и год;
- пара user–item: исторические счётчики и давность;
- косвенные связи user–genre/country/director/content_type: частоты, доли, давность;
- отношения 7/30 дней; item-item score, сходство по жанрам, fusion score и rank.

Агрегаты по категории используют первое значение списка, чтобы явно ограничить
стоимость; жанровый retrieval использует все жанры. Это упрощение документировано.
Прямые user–item признаки здесь нулевые/константные, поскольку рекомендуются новые
пары, а исходник уникален по паре. CatBoost игнорирует такие признаки; результат
не выдаётся за исследование повторного потребления. IDs не передаются как категории,
поэтому нет запоминания конкретного пользователя по идентификатору.

## 5. Исследовательская идея: ограниченная грамматика признаков

Гипотеза: стандартные 7/30 дней не оптимальны для всех связей. Алгоритм может
автоматически строить дополнительные масштабы времени и выбирать компактное
дополнение к сильному baseline. Это bounded feature synthesis, а не LLM-агент
и не утверждение о принципиально новом алгоритме автоматического ML.

Для ключа k определён счётчик
`A(k,w,t) = sum(1[key(event)=k] * 1[t-w <= timestamp < t])`.
Грамматика комбинирует семь ключей: u, i, (u,i), (u,genre), (u,country),
(u,director), (u,content_type), окна 1/3/10/14/21/60 дней и операторы:
count, сглаженная доля, отношение к 30-дневному масштабу, экспоненциальное затухание.
Half-life затухания: 3/10/30 дней,
`E(k,h,t) = sum(1[key=k] * 2^(-(t-timestamp)/h))`.
Доля категории: `A(u,c,w)/(A(u,w)+1)`; сглаженное отношение счётчиков:
`(A(k,w)+1)/(A(k,30)+30/w)`. Все промежуточные величины — только из прошлого.

После раскрытия грамматики получается 129 дополнительных столбцов, всего 193.
Пилотный CatBoost (depth=5, 200 деревьев) ранжирует дополнительные признаки по
PredictionValuesChange на **train**. Константные/нулевые по важности не включаются
в список. Проверяются префиксы top-12/top-24/top-48, каждый вместе со всеми ручными
признаками. Размер выбирается по NDCG@10 на validation-search. Выбран top-{lock['search_top_k']}.

Ограничение: поиск не перебирает произвольные SQL-программы, а важность отдельного
признака может пропустить полезные взаимодействия. Схема связей и пространство
операторов заданы человеком. Автоматизированы генерация комбинаций и отбор,
а не вся постановка задачи.

## 6. Честное сравнение

Четыре обучаемые ветки: manual; manual+all-auto; manual+selected-auto;
manual без косвенных агрегатов и retrieval scores. Последняя — абляция reranker:
персонализированный retrieval остаётся общим, поэтому это не полностью
неперсонализированная система. Дополнительно показан popularity baseline.

У всех обучаемых веток одна сетка: depth 5/7, learning_rate=0.07, l2=5,
300 деревьев, checkpoints 100/200/300. Два fit на ветку, шесть проверок качества.
Лучший checkpoint определяется по NDCG@10 только на validation-tune. Категории
обрабатывает CatBoost, 4 CPU threads, основной seed=42. У selected есть
**дополнительный бюджет поиска** (пилот + 3 fit), он явно учитывается.

{config_table}

Конфигурация сохранена в `locked_config.json` с временем и hash протокола
**до** построения тестовых признаков. `select` не читает test-файлы, а `final-test`
требует lock и отказывается перезаписывать существующую итоговую таблицу.
Контроль не является криптографическим proof предрегистрации; это локальный
журнал эксперимента и программная защита от случайного повторного подбора.

## 7. Метрики

NDCG@10 и Recall@10 усредняются по пользователям с хотя бы одним новым
положительным объектом в горизонте. Нормировка IDCG использует **все** positives,
включая не попавшие в retrieval и отсутствующие в известном каталоге:
`IDCG@10 = sum(j=1..min(10,|R_u|), 1/log2(j+1))`.
Это качество полученных top-10 относительно всей наблюдаемой релевантности,
а не NDCG на искусственном наборе «один positive + sampled negatives».
Фильтр положительных запросов применяется только при вычислении метрики,
не при построении запросов, истории или кандидатов.

{ndcg_all}

ROC-AUC и LogLoss считаются на полной общей таблице кандидатов, в том числе
для пользователей без будущих взаимодействий. Они относятся только к этой
политике retrieval и бинарной метке потребления; не измеряют калибровку CTR.
Для popularity LogLoss не приводится: raw popularity не вероятность.
Сравнение primary: selected − manual. Парный bootstrap пользователей, 2 000
повторов с seed 42; 95% percentile interval. На одном test-срезе у пользователя
один запрос, поэтому единица ресэмплирования соответствует зависимости данных.

## 8. Результаты

{results_table}

**Основное сравнение:** {verdict} Разность {delta:+.5f}, относительное изменение
{relative:+.2f}%. Все ветки показаны, но победитель не выбирается задним числом по тесту.

{seed_table}

{mean_text} Дополнительные seed не используются для выбора признаков или
параметров. Интервалы условны относительно фиксированной когорты, недели и retrieval;
они не учитывают весь разброс по датасетам, времени и процедуре model selection.
{robustness}

## 9. Что сработало и что ограничивает результат

Простой popularity baseline даёт NDCG@10 **{popular.ndcg10:.5f}**, близко к
ручному CatBoost **{base.ndcg10:.5f}**. Более сложная модель здесь не доказала
существенного превосходства над популярностью. Под «сильным baseline» понимается
содержательная инженерная конструкция и честный подбор, а не доказанный SOTA.
Глобальная AUC CatBoost выше, но её могут улучшать пользовательские признаки
вероятности активности, одинаковые для всех кандидатов пользователя. Это
не обязательно улучшает порядок объектов внутри его списка; поэтому primary — NDCG.

- Полный набор: NDCG@10 {all_auto.ndcg10:.5f}, разность с ручным baseline
  {all_auto.ndcg10-base.ndcg10:+.5f}. Больше признаков не означает лучшее обобщение.
- Отбор: NDCG@10 {primary.ndcg10:.5f}; разность с all-auto
  {primary.ndcg10-all_auto.ndcg10:+.5f}. Это описательное сравнение, без отдельного
  подтверждающего теста гипотезы; основная заранее выбранная пара — selected/manual.
- Абляция косвенных сигналов: NDCG@10 {ablated.ndcg10:.5f} против {base.ndcg10:.5f}
  у полного ручного baseline. Проверяется совместный блок сигналов, не отдельный
  жанр или окно, и персонализированный retrieval остаётся неизменным.
- LogLoss selected {primary.logloss:.6f}, manual {base.logloss:.6f}; ROC-AUC
  {primary.auc:.5f} и {base.auc:.5f}. Разные метрики могут двигаться в разные стороны:
  checkpoint выбирался по NDCG, а не по калибровке вероятностей. Утверждение
  о безусловном улучшении всех аспектов качества не делается.

Параметры каждой ветки выбирались отдельно из одинаковой сетки. Поэтому наблюдаемая
разность относится к **всей процедуре feature search + model selection**, а не
к чистому причинному эффекту добавленных столбцов при одинаковой глубине/числе деревьев.
{robustness} Более высокая вариативность может быть связана с разреженными
положительными метками, коррелированными признаками и выбранной глубиной дерева;
это объясняющие гипотезы, а не доказанные причины. Их следует проверять matched-depth
сравнением и несколькими временными фолдами, а не новым подбором на этом test.

Наиболее важные признаки итоговой selected-модели: {top_all}.

Ведущие автоматически добавленные признаки: {top_auto}.
Полный список отбора: `results/main/locked_config.json`, таблицы важности:
`*_importance.csv`. Importance не является причинным эффектом; коррелированные
окна делят важность. Ветка all-auto отделяет эффект увеличения пространства
от эффекта его отбора; no-indirect показывает совместную роль косвенных связей
и переданных retrieval scores. Индивидуальный вклад каждого окна экспериментально
не изолирован, поэтому причинные утверждения о конкретном окне не делаются.

Короткие и длинные окна дают модели разные масштабы активности и популярности.
Но полный набор увеличивает вычисления и риск подгонки, а отбор на одной неделе
может быть нестабилен при temporal drift. {verdict}

Разрезы по длине истории 1–2, 3–5, 6–20 и 21+ доступны в
`activity_strata.csv`; это описательный анализ, не новые основания для настройки.
Пропуски retrieval ограничивают максимально достижимое качество любого reranker.
Следующий содержательный эксперимент — несколько rolling-origin test-недель
с полностью вложенным подбором, более крупные пользовательские когорты,
out-of-time permutation importance и cost-aware отбор групп признаков.

## 10. Вычислительные затраты и масштабируемость

{cost_table}

Построение кандидатов на тесте: {test['candidate_seconds']:.2f} с; всех признаков:
{test['feature_seconds']:.2f} с; вся подготовка тестового среза:
{test['total_seconds']:.2f} с. Время отдельного поиска selected:
**{lock['search_seconds']:.1f} с**, сверх общей сетки финальных моделей.
Время inference не включает чтение файлов и сборку CatBoost Pool; эти составляющие
разделены в `runtime.csv`. Размер DataFrame всех тестовых признаков:
{test['feature_bytes']/1048576:.1f} MiB; это не полная память процесса.
Фактическая платформа и peak RSS сохранены в environment/audit JSON.
Замеры wall-clock сделаны на рабочем ноутбуке, без изоляции фоновой нагрузки;
они ориентировочные, разброс времени повторными прогонами не оценивался.

![Локальное масштабирование](reports/scaling.png)

`scaling.csv` содержит label-free измерения 100/500/1000 запросов на одном
историческом срезе; каталог и история фиксированы, каждый размер измерен один раз.
Это микробенчмарк числа запросов, не доказательство работы на миллиардах событий.

Генерация F агрегатов в текущей реализации повторяет groupby для каждого окна:
примерно O(W·R·H + F·Q·C), где H — история, Q — запросы, C=100, R — ключи.
Хранение признаков O(F·Q·C). При промышленном масштабе нужны инкрементальные
счётчики, оконные состояния, feature store и вычисление только выбранных столбцов.
Текущий builder всегда строит все 193 столбца; меньший selected-набор экономит
обучение/скоринг модели, **но не доказанно экономит feature generation**.

Item-item cooccurrence строится sparse-multiplication, но может уплотняться;
на больших каталогах его стоимость определяется суммой квадратов длин историй.
Прореживание top-50 выполняется после умножения и не снимает этот пиковый риск.
Поиск кандидатов в Python сортирует известный каталог по каждому пользователю,
порядка O(Q·I·log I). Для production нужны блочные top-k, ANN, distributed retrieval
и контроль активности heavy users; простое увеличение объёма здесь недостаточно.

## 11. Ограничения и честные выводы

1. Единственная случайная когорта из 10 000 пользователей и один test-горизонт.
2. KION — агрегированная implicit-feedback выгрузка; повторные сеансы потеряны,
   достоверные показы/негативы отсутствуют, возможен exposure bias. Любое
   зарегистрированное взаимодействие считается positive; удовлетворённость
   пользователя и минимальная длительность просмотра не оцениваются.
3. Условие активности за 28 дней исключает холодных и давно неактивных пользователей.
4. Известный каталог строится только по сэмплу: часть реально доступных объектов
   не попадает в него, но их будущая релевантность остаётся в знаменателе метрики.
5. Снимок item metadata считается статическим; нельзя доказать историческую
   доступность каждого поля. Демография и агрегированная длительность исключены.
6. Конечная грамматика и ограниченный поиск не доказывают оптимальность признаков.
7. Результаты нельзя напрямую сопоставлять с официальным leaderboard, другим
   retrieval, CTR-задачей или онлайн-эффектом. A/B-тест не проводился.

Практический вывод: автоматизацию нужно оценивать как добавку к сильному
воспроизводимому baseline с честным бюджетом поиска, а не заменять проверку
гипотез большим числом признаков. {verdict}

## 12. Воспроизведение

Python **3.11**, CPU, интернет только для публичных данных и PyPI. GPU и платные API
не нужны. Рекомендуется 8–16 GiB RAM; точное время зависит от CPU. Зафиксированные
версии в `requirements-lock.txt`, ключевые прямые зависимости в `requirements.txt`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pytest -q
python -m src.data --users 10000
python -u -m src.experiment build-dev
python -u -m src.audit
python -u -m src.experiment select
python -u -m src.experiment final-test
python -u -m src.audit --include-test
python -u -m src.benchmark
python -m src.report
```

Альтернатива после установки зависимостей: `PYTHON=.venv/bin/python bash run.sh`.
В поставке есть уже рассчитанные результаты. Для **полного нового прогона** сначала
сохраните их под другим именем: `mv results/main results/main_reference`.
Новый запуск создаст `results/main`; исходные данные при повторном запуске не
скачиваются заново, а проверяются checksum. Небольшие расхождения времени и
floating-point результатов между платформами возможны; фиксированный seed не
обещает побитовую идентичность разных CPU/версий. Архив содержит модели, метрики,
логи и per-user результаты; сырые данные и тяжёлые feature caches скачиваются/
создаются командами выше.

Для иной когорты: `python -m src.data --users 5000`, затем те же три стадии
`src.experiment` с `--users 5000`. Это новый эксперимент; его результаты здесь
не заявляются. `src.report` и query-scaling benchmark оформляют основной 10k run.

### Структура

- `PROTOCOL.md`: протокол до открытия теста.
- `src/data.py`: загрузка, проверки, полные истории пользователей.
- `src/features.py`: retrieval и point-in-time признаки, отдельное добавление labels.
- `src/experiment.py`: подбор, фиксация, финальная оценка.
- `src/metrics.py`: метрики и парный bootstrap.
- `tests/test_temporal.py`: проверки корректности на контролируемых данных.
- `results/main/`: trials, configuration lock, метрики, статистика срезов, модели.
- `reports/TLab_Report.pdf`: краткая версия этого отчёта; графики рядом.

### Соответствие четырём требованиям задания

| Требование | Реализация | Доказательство |
| --- | --- | --- |
| Сильный baseline | CatBoost, 64 ручных признака, временные срезы | Код, тесты, manual-строка метрик |
| Исследовательская идея | Грамматика 129 новых агрегатов + автоматический top-k отбор | pilot_importance, search_trials, lock |
| Сравнение | Общие кандидаты, две validation-недели, закрытый test, 3 seeds | Hash кандидатов, tuning_trials, test_metrics, bootstrap |
| Анализ | All-auto/no-indirect абляции, importance, разрезы, стоимость и ограничения | Таблицы, графики, scaling.csv, разделы 9–11 |

## 13. Источники и использование инструментов

1. Petrov et al. **MTS Kion Implicit Contextualised Sequential Dataset for Movie
   Recommendation** (2022). [Статья](https://arxiv.org/abs/2209.00325),
   [официальные данные](https://github.com/irsafilo/KION_DATASET).
2. Prokhorenkova et al. **CatBoost: unbiased boosting with categorical features**
   (NeurIPS 2018). [Статья](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html).
   [API](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier).
3. Kanter, Veeramachaneni. **Deep Feature Synthesis: Towards Automating Data Science
   Endeavors** (2015). [Статья MIT](https://dai.lids.mit.edu/wp-content/uploads/2017/10/DSAA_DSM_2015.pdf).
   Связь с работой: ограниченная композиция агрегатов по реляционным ключам;
   здесь дополнительно явно задаются временные границы и валидационный отбор.
4. Krichene, Rendle. **On Sampled Metrics for Item Recommendation** (KDD 2020).
   [Страница авторов](https://research.google/pubs/on-sampled-metrics-for-item-recommendation/).
   В этой работе не используется оценка на случайно выбранных negatives: оценивается
   фиксированная retrieval-политика с полным знаменателем наблюдаемых positives.

Codex использован для реализации, тестирования и подготовки отчёта, что разрешено
условием задания. Эксперименты выполнены локально, результаты не вымышлены.
Перед защитой необходимо понимать определения target, временные границы,
ограничения KION и смысл доверительного интервала. Отправка в T Lab не выполнялась. Данные остаются у исходного
правообладателя, в архив решения не включены; условия их использования следует
проверять у источника при дальнейшей публикации.
'''
    (reports/'EXPERIMENT.md').write_text(readme.replace('](reports/', ']('),encoding='utf-8')

    # Compact PDF, explicit page boundaries keep tables and figures readable.
    regular=font_manager.findfont('DejaVu Sans')
    bold=font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans',weight='bold'))
    pdfmetrics.registerFont(TTFont('DV',regular));pdfmetrics.registerFont(TTFont('DVB',bold))
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyRU',fontName='DV',fontSize=9.4,leading=14,spaceAfter=8,textColor=colors.HexColor('#25364b')))
    styles.add(ParagraphStyle(name='TitleRU',fontName='DVB',fontSize=26,leading=32,spaceAfter=18,textColor=colors.HexColor('#153b53')))
    styles.add(ParagraphStyle(name='HeadRU',fontName='DVB',fontSize=16,leading=21,spaceAfter=13,textColor=colors.HexColor('#153b53')))
    styles.add(ParagraphStyle(name='SmallRU',fontName='DV',fontSize=7.2,leading=10,spaceAfter=5))
    story=[]
    def p(text,style='BodyRU'): story.append(Paragraph(escape(text).replace('\n','<br/>'),styles[style]))
    def head(text): p(text,'HeadRU')
    def tbl(headers,rows,widths=None):
        data=[[Paragraph(escape(str(v)),styles['SmallRU']) for v in headers]]+[[Paragraph(escape(str(v)),styles['SmallRU']) for v in r] for r in rows]
        t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eff4')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),
            ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#98aeba')),
            ('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#dce3e9'))]))
        story.append(t);story.append(Spacer(1,10))
    def page(): story.append(PageBreak())
    p('T LAB / AI RESEARCH · ИССЛЕДОВАТЕЛЬСКОЕ ЗАДАНИЕ','SmallRU')
    story.append(Spacer(1,20))
    p('Можно ли автоматизировать\nпризнаки для рекомендаций?','TitleRU')
    p('Временная грамматика агрегатов и отбор признаков против сильного CatBoost на KION.')
    p('Карина Шаля · 17 октября 2026','SmallRU')
    story.append(Spacer(1,15))
    head('Результат в одном абзаце')
    p(f'{verdict} NDCG@10: {base.ndcg10:.5f} → {primary.ndcg10:.5f}; разность {delta:+.5f} ({relative:+.2f}%). 95% CI: [{ci["42"]["ci95_low"]:+.5f}; {ci["42"]["ci95_high"]:+.5f}].')
    p(robustness)
    p(f'10 000 пользователей; {manifest["rows"]:,} событий; 600 000 train-пар; 64 ручных признака, 193 в полном наборе, {len(selected)} после отбора. Две validation-недели, один закрытый test, три seed основного сравнения.')
    story.append(Image(str(reports/'quality.png'),width=172*mm,height=77*mm))
    p('Код, полный технический отчёт и все промежуточные таблицы — в сопровождающем архиве. Баллы задания и результат официального leaderboard не заявляются.','SmallRU')
    page();head('1 / Данные и временной протокол')
    p('KION: 5 476 251 строка, 962 179 пользователей, 15 706 объектов. Проверены реальные даты 13.03–22.08.2021. Использован полный data_en.zip и русские item metadata из зафиксированной версии официального репозитория.')
    p('Отдельный interactions.csv.zip оказался обрезанным (1 594 787 строк, последняя дата «20»). Он исключён. Загрузчик проверяет hash, CRC, число строк, даты и уникальность пар.')
    p('10 000 пользователей выбраны случайно среди известных до 12 июля. Сохранены все их строки. История признаков строго timestamp < t; target — новое зарегистрированное взаимодействие в [t,t+7 дней). Без показов это не CTR, а прогноз implicit feedback.')
    tbl(['Срез','Начало','Конец','Запросы','Пары'],[[r[0],r[1],r[2],r[3],r[4]] for r in split_rows],[28*mm,31*mm,31*mm,31*mm,31*mm])
    p('Запросы: пользователь активен в последние 28 дней. Train ограничен 2 000 запросами на срез; validation/test содержат все подходящие запросы. Будущая активность не определяет включение пользователя. Истории не усекаются.')
    p('KION агрегирован по user–item с last_watch_dt. Более ранние повторные просмотры не восстановимы. total_dur и watched_pct исключены. Корректность времени доказана относительно опубликованных псевдособытий; истинный онлайн-журнал из этих данных восстановить нельзя.')
    p('Статические item metadata считаются доступными при появлении объекта. Демография и тексты исключены. Модель обучается только на train; обновление истории событиями до test-cutoff допустимо.')
    page();head('2 / Кандидаты и ручной baseline')
    p('Для каждого среза известный каталог определяется только прошлой историей когорты. Просмотренные пары исключены. Четыре канала: популярность за 7 дней, популярность за всё время, item-item cosine и жанровая близость с поправкой на популярность.')
    p('Cosine вычисляется из бинарной user–item матрицы, со shrinkage co_count/(co_count+10), до 50 соседей на объект. Жанровый канал использует все жанры. Reciprocal-rank fusion sum(1/(60+rank)) оставляет 100 кандидатов. Ties детерминированы item_id.')
    p('Будущие правильные ответы не добавляются в кандидаты. Метки присоединяются после retrieval и features. Все методы используют одну сохранённую таблицу кандидатов с SHA-256-подписью.')
    p(notes)
    p(f'Потолок NDCG@10 при идеальном ранжировании фиксированных кандидатов: {oracle["ndcg10"]:.5f}. Это диагностическая граница с известными labels, а не обученный метод.')
    head('64 признака сильного baseline')
    p('Пользователь: частоты 7/30 дней и all-time, давность, разнообразие объектов и жанров, активные дни. Объект: частоты, доли, давность, статические категории и год. Косвенные связи: user–genre/country/director/content_type, частоты, доли и давность. Также отношения 7/30, CF score, жанровый score, fusion score и rank.')
    p('Прямые user–item признаки сохранены для общей схемы, но константны в задаче новых объектов. Для агрегатов категорий берётся первое значение списка; это явное упрощение. Идентификаторы не используются как категории модели.')
    p('CatBoostClassifier, LogLoss, все 600 000 пар, без oversampling и class weights. Нулевые метки — отсутствие зарегистрированного просмотра, а не подтверждённая нерелевантность.')
    page();head('3 / Автоматическая генерация и отбор')
    p('Гипотеза: разные связи требуют разных масштабов времени. Ограниченная грамматика строит комбинации ключей, окон и операторов. Это автоматизация внутри заданного человеком пространства, а не универсальный SQL-агент.')
    p('Ключи: user, item, user–item и четыре косвенных связи. Дополнительные окна: 1, 3, 10, 14, 21, 60 дней. Операторы: count, сглаженная доля, отношение к 30 дням, экспоненциальное затухание с half-life 3, 10, 30 дней.')
    p('A(k,w,t) = сумма индикаторов key=k и t-w ≤ timestamp < t.\nE(k,h,t) = сумма 2^(-(t-timestamp)/h) для key=k и timestamp < t.\nДоля: A(u,c,w)/(A(u,w)+1).\nОтношение: (A(k,w)+1)/(A(k,30)+30/w).')
    p(f'129 новых признаков, 193 всего. Пилотный CatBoost на train ранжирует extras по PredictionValuesChange. На validation-search сравниваются top-12/24/48 поверх всех ручных признаков. Выбран top-{lock["search_top_k"]}, итог {len(selected)} столбцов.')
    tbl(['Top-k','Признаков','Search NDCG@10'],[[int(r['size']),int(r.n_features),f'{r.ndcg10:.5f}'] for _,r in search.iterrows()],[35*mm,45*mm,70*mm])
    p('Важность на train не использует тест, но может плохо оценивать коррелированные окна. Префиксный отбор не перебирает все подмножества. Улучшение над baseline не гарантируется.')
    p('Сравнения: manual; all-auto; selected-auto; абляция manual без косвенных признаков и retrieval scores. У абляции остаётся тот же персонализированный retrieval, поэтому она проверяет reranker, не всю персонализацию.')
    page();head('4 / Подбор и измерение качества')
    p('Для каждой обучаемой ветки: depth 5/7, learning_rate 0.07, l2 5, 300 деревьев; checkpoints 100/200/300 выбираются по NDCG@10 на validation-tune. Общий бюджет — два fit и шесть checkpoints на ветку. У selected есть дополнительный бюджет feature search.')
    tbl(['Метод','F','Depth','Trees','Tune NDCG'],configs,[66*mm,15*mm,18*mm,21*mm,32*mm])
    p('locked_config.json сохранён до построения теста, содержит время и hash протокола. Повторный запуск итоговой оценки поверх результата блокируется. Это локальный audit trail, не внешняя регистрация исследования.')
    p('Primary NDCG@10: среднее по пользователям с будущими positives. IDCG строится по всем наблюдаемым positives, включая отсутствующие в retrieval и каталоге. Recall@10 имеет тот же полный знаменатель. Запросы без positives не удаляются из данных: для них дополнительно сообщается NDCG=0 при усреднении по всем запросам.')
    p('AUC и LogLoss вычисляются на общей полной таблице кандидатов. Они условны относительно retrieval-политики и не являются метриками CTR. У popularity нет LogLoss: её score не вероятность.')
    p('Primary comparison selected − manual; парный bootstrap по пользователям, 2 000 повторов, 95% percentile CI. Дополнительные seed 17/2026 используют уже выбранные параметры и признаки, без новой настройки.')
    p('10 автоматических тестов проверяют границы времени и окон, неизменность при добавлении будущих событий, кандидатов без future leakage, полные истории, знаменатели ranking-метрик, ties и bootstrap.')
    page();head('5 / Итоговые результаты')
    tbl(['Метод','F','NDCG@10','Recall@10','AUC','LogLoss'],rows,[57*mm,13*mm,25*mm,25*mm,23*mm,25*mm])
    p(verdict)
    tbl(['Seed','Manual','Selected','Разность','95% CI'],seed_rows,[15*mm,25*mm,25*mm,26*mm,69*mm])
    p(mean_text)
    p(robustness)
    p(ndcg_all)
    p('Все ветки показаны независимо от направления результата. Основное сравнение фиксировано до теста; тестовая таблица не используется для нового выбора метода. Интервалы условны относительно этой недели, когорты и retrieval, а не всей целевой популяции.')
    page();head('6 / Интерпретация результатов')
    p(f'Все автопризнаки: {all_auto.ndcg10:.5f}; без косвенных сигналов: {ablated.ndcg10:.5f}. LogLoss selected {primary.logloss:.6f}, manual {base.logloss:.6f}: выбор по NDCG не гарантирует улучшение калибровки. Ветки имеют независимо подобранные depth/trees, поэтому эффект относится ко всей процедуре, а не только к новым столбцам.')
    p(f'Popularity ({popular.ndcg10:.5f}) близка к manual ({base.ndcg10:.5f}). Сложность baseline не равна доказанному SOTA; высокая глобальная AUC может отражать прогноз активности пользователя, не лучший порядок его объектов.')
    p('Наиболее важные признаки selected-модели: '+top_all)
    p('Ведущие автоматически добавленные: '+top_auto)
    p('Feature importance не доказывает причинный вклад отдельного окна. All-auto контролирует расширение пространства, no-indirect — совместную роль косвенных сигналов. Разрезы по длине истории сохранены в activity_strata.csv.')
    story.append(Image(str(reports/'importance.png'),width=170*mm,height=76*mm))
    p('Зелёным показаны автоматически добавленные признаки, синим — ручные. Важности вычислены на обучающей выборке; тестовые метки при объяснении модели не используются.','SmallRU')
    page();head('7 / Стоимость и масштабируемость')
    tbl(['Метод','2 fit, c','Inference, c','MiB'],costs,[72*mm,30*mm,30*mm,27*mm])
    p(f'Поиск выбранных признаков: {lock["search_seconds"]:.1f} с сверх финального подбора. На тесте retrieval {test["candidate_seconds"]:.2f} с, все features {test["feature_seconds"]:.2f} с; DataFrame {test["feature_bytes"]/1048576:.1f} MiB. Inference в таблице исключает сборку Pool и чтение файлов.')
    story.append(Image(str(reports/'scaling.png'),width=170*mm,height=60*mm))
    p('Микробенчмарк меняет число запросов при фиксированной истории; один замер на размер. Он не доказывает масштабирование числа событий. 4 CPU threads, GPU не использовался. Платформа и peak RSS сохранены в JSON.')
    p('Features: примерно O(W·R·H + F·Q·C), память O(F·Q·C). Сейчас builder строит все 193 признака даже для selected; экономия feature-generation не заявляется. Для production нужны инкрементальные окна и вычисление только выбранных столбцов.')
    p('Sparse cooccurrence может уплотняться; его стоимость зависит от суммы квадратов длин историй. Top-50 pruning выполняется после умножения. Candidate retrieval сортирует каталог для каждого пользователя, примерно O(Q·I·log I). Для больших данных требуются ANN/блочные top-k и распределённые состояния.')
    page();head('8 / Ограничения и воспроизведение')
    p('Одна когорта и одна тестовая неделя; неизвестные показы и exposure bias; потерянные повторные сеансы; статические непроверяемые во времени metadata; неизвестные каталогу объекты; ограниченная грамматика и validation overfitting. Онлайн-эффект и перенос на Yambda не измерялись.')
    p('Следующие эксперименты: rolling-origin оценка на нескольких неделях с вложенным подбором, более крупные когорты, стабильность отбора по времени и cost-aware selection групп. Улучшение в другой метрике не заменяет заранее выбранную NDCG@10.')
    head('Запуск')
    p('Python 3.11, виртуальное окружение, pip install -r requirements-lock.txt.\npython -m pytest -q\npython -m src.data --users 10000\npython -m src.experiment build-dev\npython -m src.experiment select\npython -m src.experiment final-test\npython -m src.benchmark\npython -m src.report','SmallRU')
    p('Для нового прогона сохраните results/main как results/main_reference. Сырые данные скачиваются публично, проверяются по SHA-256. Код, конфигурации, модели, логи, таблицы и per-user результаты в архиве; cache и сырые данные не включены. Полные пояснения — README.md.')
    head('Источники')
    for text,url in [
        ('Petrov et al. MTS Kion Dataset (2022)','https://arxiv.org/abs/2209.00325'),
        ('Prokhorenkova et al. CatBoost (NeurIPS 2018)','https://arxiv.org/abs/1706.09516'),
        ('Kanter, Veeramachaneni. Deep Feature Synthesis (2015)','https://dai.lids.mit.edu/wp-content/uploads/2017/10/DSAA_DSM_2015.pdf'),
        ('Krichene, Rendle. On Sampled Metrics (KDD 2020)','https://research.google/pubs/on-sampled-metrics-for-item-recommendation/')]:
        story.append(Paragraph(f'<link href="{url}" color="#147d6f">{escape(text)}</link>',styles['SmallRU']))
    p('Codex использован для кода, проверок и отчёта, как разрешено заданием. Все метрики рассчитаны локально. Репозиторий: github.com/KarinaSH7/kion-temporal-features.','SmallRU')
    def footer(canvas,doc):
        canvas.setFont('DV',8);canvas.setFillColor(colors.HexColor('#6d8090'))
        canvas.drawString(20*mm,13*mm,'T LAB · TEMPORAL FEATURE SEARCH · KION')
        canvas.drawRightString(190*mm,13*mm,str(doc.page))
    doc=SimpleDocTemplate(str(reports/'TLab_Report.pdf'),pagesize=(210*mm,297*mm),rightMargin=20*mm,leftMargin=20*mm,
                          topMargin=18*mm,bottomMargin=22*mm,title='T Lab: автоматический поиск временных признаков',author='Карина Шаля')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print('Created reports/EXPERIMENT.md, reports/TLab_Report.pdf and charts',flush=True)


if __name__=='__main__': main()

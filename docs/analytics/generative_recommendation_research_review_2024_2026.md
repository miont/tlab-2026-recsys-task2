# Новые исследования по Generative Recommendation и Reinforcement Learning (2024–2026)

**Дата обзора:** 22 октября 2026 г.  
**Базовая работа для сравнения:** Petrov A. V., Macdonald C., *Aligning GPTRec with Beyond-Accuracy Goals with Reinforcement Learning* (2024), [arXiv:2403.04875](https://arxiv.org/abs/2403.04875).

## 1. Область исследования

В обзоре представлены более новые исследования, развивающие идеи GPTRec-RL: последовательная генерация рекомендаций, оптимизация качества полного списка (list-wise optimization), reinforcement learning (PPO, GRPO и их варианты), preference alignment (DPO), а также оптимизация разнообразия, новизны, fairness и popularity bias. Отдельно учитываются наличие исходного кода и сложность воспроизведения.

## 2. Краткая сравнительная таблица

| № | Работа | Год | Основной метод | Близость к GPTRec-RL | Код |
|---|---|---|---|---|---|
| 1 | **BLADE: Bayesian List-wise Alignment** | 2026 | Bayesian list-wise alignment: NDCG, fairness, diversity | Очень высокая | [GitHub](https://github.com/RegionCh/BLADE) |
| 2 | **MiniOneRec** | 2025 | Semantic IDs + SFT + GRPO | Очень высокая | [GitHub](https://github.com/AkaliKong/MiniOneRec) |
| 3 | **LAAC** | 2025 | LLM-guided adversarial actor–critic, diversity/novelty | Очень высокая | Не подтверждён |
| 4 | **V-STAR** | 2026 | Value-guided decoding + Sibling-GRPO | Высокая | [GitHub](https://github.com/Yara-HYR/V-star) |
| 5 | **OneRec** | 2025 | Generative recommender + DPO | Высокая | Требует проверки |
| 6 | **Align³GR** | 2026 | Multi-level alignment + DPO | Высокая | Не подтверждён |
| 7 | **DiffuReason** | 2026 | Think-then-Diffuse + GRPO | Средняя–высокая | Не подтверждён |
| 8 | **BASRec** | 2025 | Аугментация последовательностей для relevance/diversity | Средняя | [GitHub](https://github.com/KingGugu/BASRec) |

**Примечание:** близость и практическая сложность — аналитические оценки; они не измерялись авторами по единому протоколу. «Не подтверждён» не означает, что код заведомо отсутствует.

## 3. BLADE — Bayesian List-wise Alignment (SIGIR 2026)

**Название:** *Beyond Static Best-of-N: Bayesian List-wise Alignment for LLM-based Recommendation*  
**Авторы:** Ruijun Chen, Chongming Gao, Jiawei Chen, Weiqin Yang, Xiangnan He.  
**Источники:** [arXiv:2605.04559](https://arxiv.org/abs/2605.04559), [PDF](https://arxiv.org/pdf/2605.04559), [GitHub](https://github.com/RegionCh/BLADE).

### Методика

BLADE непосредственно оптимизирует метрики всего списка, включая NDCG, diversity и fairness. Обычная токенная кросс-энтропия обучает предсказывать следующий токен, но не гарантирует хороших свойств полного набора рекомендаций:

\[
L=-\sum_t\log P_\theta(i_t\mid H,i_{<t}).
\]

Подход опирается на **Best-of-N alignment**: генерация нескольких рекомендательных списков, оценка каждого целевой метрикой, отбор предпочтительных вариантов и обновление модели. В отличие от статического Best-of-N, используется байесовский механизм обновления целевого распределения с учётом предыдущих наблюдений и новых генераций.

### Сравнение с GPTRec-RL

| Аспект | GPTRec-RL | BLADE |
|---|---|---|
| Базовая модель | GPTRec | LLM-based recommender |
| Оптимизация | PPO (Actor–Critic) | Bayesian list-wise alignment |
| Единица оптимизации | Последовательность рекомендаций | Полный список |
| Beyond-accuracy | Diversity, popularity bias | Diversity, fairness |
| Отдельный PPO-Critic | Да | Нет |

**Практический интерес:** возможность сравнить `GPTRec + PPO` и `GPTRec + BLADE` на одинаковых датасетах и функциях качества. Приоритет для изучения — **очень высокий**.

## 4. MiniOneRec (2025)

**Название:** *MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation*  
**Источники:** [arXiv:2510.24431](https://arxiv.org/abs/2510.24431), [PDF](https://arxiv.org/pdf/2510.24431), [GitHub](https://github.com/AkaliKong/MiniOneRec).

### Методика

В отличие от GPTRec, где объект соответствует отдельному item ID, MiniOneRec использует **Semantic IDs (SID)** — последовательности дискретных токенов, полученных из семантических представлений объектов с использованием методов остаточного квантования (например, RQ-VAE).

Пример представления объекта:

```text
Movie: Interstellar
Semantic ID: <sid_17><sid_83><sid_5>
```

Pipeline:

```text
Semantic ID Construction
        ↓
Supervised Fine-Tuning (SFT)
        ↓
GRPO Fine-Tuning
        ↓
Evaluation
```

На RL-этапе применяется **Group Relative Policy Optimization (GRPO)**: для одного запроса генерируются несколько кандидатов, затем их награды сравниваются внутри группы. Упрощённая оценка преимущества:

\[
A_j=\frac{R_j-\mu_R}{\sigma_R+\varepsilon}.
\]

Этот принцип позволяет отказаться от отдельной value-модели, характерной для PPO Actor–Critic.

### Сравнение с GPTRec-RL

| Аспект | GPTRec-RL | MiniOneRec |
|---|---|---|
| Основа | Архитектура GPT-2 | Qwen |
| Объекты | Item IDs | Semantic IDs |
| Предобучение | Teacher–Student | SFT/semantic alignment |
| RL | PPO | GRPO |
| Отдельный Critic | Да | Нет |
| Цели | Beyond-accuracy | Scaling, accuracy, candidate diversity |

**Ограничение для Colab:** полное воспроизведение может требовать нескольких A100/H100; это не тот же уровень затрат, что небольшой GPTRec. Кроме того, *candidate diversity* нельзя без дополнительной проверки отождествлять с ILD@10.

**Практический интерес:** современный открытый pipeline генеративного рекомендателя с RL; высокий приоритет.

## 5. LAAC (2025)

**Название:** *Large Language Model-Enhanced Reinforcement Learning for Diverse and Novel Recommendations*  
**Авторы:** Jiin Woo, Alireza Bagheri Garakani, Tianchen Zhou, Zhishen Huang, Yan Gao.  
**Источники:** [arXiv:2507.21274](https://arxiv.org/abs/2507.21274), [PDF](https://arxiv.org/pdf/2507.21274).

### Методика

LAAC (LLM-guided Adversarial Actor Critic) направлена непосредственно на **diversity** и **novelty**. Проблема обычного RL — неэффективное исследование пространства рекомендаций: случайные новые объекты часто нерелевантны.

LLM используется как внешняя reference policy, предлагающая перспективные новые объекты; основная модель Actor–Critic обучается выбирать рекомендации. Critic получает дополнительные механизмы регуляризации оценок для слабо исследованных действий.

```text
User History → LLM Reference → Novel Candidates
                                  ↓
                           Actor–Critic
                                  ↓
                            Recommendations
```

**Отличие от GPTRec-RL:** акцент не только на настройке награды, но и на направленном исследовании множества потенциально релевантных объектов. Официальная открытая реализация в исходном обзоре не была подтверждена.

**Практический интерес:** высок для исследований novelty и diversity без полного переобучения крупной LLM.

## 6. V-STAR (KDD 2026)

**Название:** *Spend Search Where It Pays: Value-Guided Structured Sampling and Optimization for Generative Recommendation*  
**Источники:** [arXiv:2602.10699](https://arxiv.org/abs/2602.10699), [PDF](https://arxiv.org/pdf/2602.10699), [GitHub](https://github.com/Yara-HYR/V-star).

### Методика

V-STAR фокусируется на эффективности исследования дерева генерации и качестве credit assignment в RL:

- **Value-Guided Efficient Decoding (VED)** направляет вычисления на перспективные ветви генерации.
- **Sibling-GRPO** сравнивает альтернативные действия из общего родительского узла, а не только награды целых генераций.

```text
User History
  ├─ Item A
  │    ├─ Item B
  │    └─ Item C
  └─ Item D
       ├─ Item E
       └─ Item F
```

**Отличие от GPTRec-RL:** более детальный учёт того, какие варианты исследуются и каким шагам приписывается полезность. Авторский репозиторий опубликован, но для полного воспроизведения заявлены высокие требования к GPU (например, 8 × A100 80 GB).

**Практический интерес:** высокий при изучении современных sampling-алгоритмов и credit assignment.

## 7. OneRec (2025)

**Название:** *OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment*  
**Источники:** [arXiv:2502.18965](https://arxiv.org/abs/2502.18965), [PDF](https://arxiv.org/pdf/2502.18965).

### Методика

OneRec предлагает объединить традиционные этапы retrieval и ranking в единую генеративную модель. Для последующего согласования предпочтений используется **Direct Preference Optimization (DPO)**.

Вместо явной оптимизации скалярной награды модель обучается предпочитать списки \(G^+\) спискам \(G^-\). Существенная задача — сформировать пары предпочтений, поскольку в пользовательских логах обычно наблюдается отклик только на показанные рекомендации. Для этого используются reward modeling и процедуры формирования данных предпочтений.

**Сравнение с GPTRec-RL:** DPO позволяет избежать отдельного PPO-Critic, но предъявляет требования к качеству пар предпочтений. В работе сообщается об улучшении watch-time на 1,6% в промышленном A/B-эксперименте Kuaishou; это не сопоставимо напрямую с offline NDCG.

**Практический интерес:** высокий как альтернатива PPO.

## 8. Align³GR (AAAI 2026)

**Название:** *Align³GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation*  
**Источники:** [arXiv:2511.11255](https://arxiv.org/abs/2511.11255), [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/38651).

### Методика

Согласование на трёх уровнях:

1. **Token-level alignment:** текст и рекомендательные идентификаторы объектов.
2. **Behavior-level alignment:** истории взаимодействий пользователей.
3. **Preference-level alignment:** предпочтения, оптимизируемые через DPO и последующие итерации.

Авторы сообщают об улучшении относительно выбранных baseline: Recall@10 на 17,8% и NDCG@10 на 20,2%, а также приводят онлайн-результаты. Эти проценты привязаны к конкретному протоколу оригинального исследования.

**Практический интерес:** понимание перехода от небольшого GPTRec к многоуровневым LLM-рекомендателям; читать после MiniOneRec и OneRec.

## 9. DiffuReason (2026)

**Название:** *DiffuReason: Bridging Latent Reasoning and Generative Refinement for Sequential Recommendation*  
**Источники:** [arXiv:2602.09744](https://arxiv.org/abs/2602.09744), [PDF](https://arxiv.org/pdf/2602.09744).

### Методика

Архитектура **Think-then-Diffuse**:

- **Think:** формирование скрытого представления предпочтений пользователя с помощью thinking tokens.
- **Diffuse:** итеративное уточнение представления в diffusion-процессе.
- **GRPO:** совместная оптимизация компонентов.

В отличие от GPTRec-RL, акцент на качестве последовательной рекомендации, а не явной оптимизации ILD/popularity bias.

**Практический интерес:** перспективная смежная архитектура, но не первоочередная для прямой репликации GPTRec-RL.

## 10. BASRec (AAAI 2025)

**Название:** *Augmenting Sequential Recommendation with Balanced Relevance and Diversity*  
**Источники:** [arXiv:2412.08300](https://arxiv.org/abs/2412.08300), [PDF](https://arxiv.org/pdf/2412.08300), [GitHub](https://github.com/KingGugu/BASRec).

### Методика

BASRec решает задачу баланса relevance/diversity через **аугментацию обучающих последовательностей**, а не через PPO:

- **Single-sequence Augmentation** создаёт новые варианты истории одного пользователя.
- **Cross-sequence Augmentation** комбинирует представления разных последовательностей, стараясь не разрушать значимую семантику.

Потенциальное комбинирование с GPTRec-RL:

```text
BASRec Data Augmentation
         ↓
Supervised GPTRec Training
         ↓
PPO / GRPO Fine-Tuning
```

**Практический интерес:** сильный baseline для улучшения разнообразия без тяжёлого RL.

## 11. Дополнительные работы

### 11.1. ReRe (2025)

**Название:** *Reinforced Preference Optimization for Recommendation*  
**Источник:** [arXiv:2510.12211](https://arxiv.org/abs/2510.12211).

Исследует адаптацию RL with Verifiable Rewards к генеративным рекомендациям: constrained beam search, борьбу с дублирующимися или недопустимыми объектами и более информативное конструирование ranking rewards. Интересна для изучения декодирования, sampling и reward modeling.

### 11.2. CatDive (2026)

**Название:** *CatDive: A simple yet effective method for maximizing category diversity in sequential recommendation*  
**Источники:** [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0341419), [GitHub](https://github.com/snudatalab/CatDive).

Работа направлена на категориальное разнообразие последовательных рекомендаций без тяжёлого RL pipeline. Полезна как доступный baseline для проверки, оправдывает ли PPO дополнительные вычислительные расходы.

### 11.3. GAPO (EMNLP 2025)

**Название:** *Group-Aware Reinforcement Learning for Output Diversity in Large Language Models*  
**Источник:** [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1649/).

GAPO расширяет идеи групповой RL-оптимизации, учитывая разнообразие набора генераций. Это **не** специализированная работа о рекомендациях: чтобы использовать её для ILD@K, надо различать межгенерационное разнообразие и разнообразие объектов внутри одного списка.

## 12. Итоговое сравнение методов

| Работа | Оптимизация | List-wise | Beyond-accuracy | Сложность воспроизведения* |
|---|---|---|---|---|
| GPTRec-RL (2024) | Teacher–Student + PPO | Да | Diversity, popularity | Средняя |
| BLADE (2026) | Bayesian alignment | Да | Diversity, fairness | Средняя–высокая |
| MiniOneRec (2025) | SFT + GRPO | Частично/в зависимости от reward | Candidate diversity | Высокая |
| LAAC (2025) | LLM-guided Actor–Critic | Не основной акцент | Novelty, diversity | Средняя–высокая |
| V-STAR (2026) | SFT + Sibling-GRPO | Через дерево генераций | Candidate diversity | Высокая |
| OneRec (2025) | Generative SFT + DPO | Да, session-wise | Не основной акцент | Высокая |
| Align³GR (2026) | Multi-level alignment + DPO | Через preference alignment | Не основной акцент | Высокая |
| DiffuReason (2026) | Diffusion + GRPO | Не основной акцент | Косвенно | Высокая |
| BASRec (2025) | Data augmentation | Нет | Diversity представлений | Средняя |
| CatDive (2026) | Diversity-oriented recommendation | Категориальное разнообразие | Diversity | Относительно низкая |

\* Оценки носят качественный характер.

## 13. Основные тенденции 2024–2026

**От PPO к GRPO, DPO и list-wise alignment.** Появляется больше вариантов оптимизации сложных критериев без отдельной value-модели. Это не гарантирует уменьшение общего времени обучения: стоимость генерации и оценки кандидатов остаётся важной.

**От Item IDs к Semantic IDs.** Модели используют дискретные семантические коды объектов; улучшается работа с семантикой, но усложняются построение кодов и ограниченное декодирование.

**Улучшение sampling и credit assignment.** Работы вроде ReRe и V-STAR показывают, что оптимизация reward недостаточна, если алгоритм не исследует качественные альтернативы.

**Прямая оптимизация свойств полного списка.** BLADE развивает обучение под NDCG, diversity, fairness и другие недифференцируемые list-wise метрики.

## 14. План практического исследования

### Направление A — небольшая GPTRec в Google Colab

Цель — сравнить алгоритмы оптимизации на одинаковой архитектуре, данных и вычислительном бюджете.

1. Воспроизвести **GPTRec Supervised** на MovieLens-1M.
2. Воспроизвести **GPTRec + PPO**.
3. Реализовать экспериментальный **GPTRec + GRPO** без Critic.
4. Сформировать пары предпочтений и протестировать **GPTRec + DPO**.
5. Протестировать адаптацию **list-wise alignment**.
6. Сопоставить с **BERT4Rec + MMR**, BASRec/CatDive при совместимых протоколах.

Метрики:

\[
NDCG@10,\quad Recall@10,\quad ILD@10,\quad Coverage@10,
\]

а также novelty и метрики популярности. Сравнение проводить по Pareto frontier (`accuracy vs diversity`), фиксируя GPU-время, количество генераций, seed и настройки декодирования.

Главный исследовательский вопрос: **дают ли GRPO/DPO/list-wise методы преимущества относительно PPO при одинаковой модели и бюджете?**

### Направление B — масштабируемые LLM-рекомендатели

Рекомендуемый порядок чтения и экспериментов:

1. **MiniOneRec:** Semantic IDs, SFT, GRPO.
2. **ReRe:** ограничения декодирования, sampling, reward design.
3. **V-STAR:** tree search и credit assignment.
4. **BLADE:** непосредственная list-wise оптимизация.

Полный запуск LLM-вариантов может потребовать нескольких мощных GPU и выходит за пределы типичного бесплатного Colab.

## 15. Итоговый рейтинг

| Приоритет | Работа | Причина |
|---|---|---|
| **1** | **BLADE** | Ближайшая постановка: NDCG, diversity, fairness всего списка |
| **2** | **MiniOneRec** | Открытая современная реализация генеративных рекомендаций с GRPO |
| **3** | **LAAC** | Прямая оптимизация novelty/diversity с LLM-guided RL |
| **4** | **ReRe** | Reward modeling и sampling для генеративных рекомендаций |
| **5** | **V-STAR** | Sibling-GRPO и структурированный поиск |
| **6** | **OneRec** | DPO как альтернативный путь preference alignment |
| **7** | **CatDive** | Простой практически полезный baseline diversity |

### Заключение

Для глубокого изучения стоит начать с **BLADE**, **MiniOneRec** и **LAAC**. Для экономного экспериментального сравнения в Google Colab наиболее рационально сохранить небольшую архитектуру GPTRec и последовательно сравнить PPO с адаптированными GRPO/DPO и list-wise методами. Это позволит изолировать влияние обучающего алгоритма от изменений архитектуры и датасета.

---

**Примечание о проверке:** данный документ сохраняет содержание предыдущего обзора. Публикационные данные, ссылки и наличие кода приведены в соответствии с ним; независимая повторная проверка всех репозиториев и заявленных метрик при экспорте не выполнялась.

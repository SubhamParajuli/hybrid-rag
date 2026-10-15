examples = [
    (
        {
            "question": "What is the name of the network architecture proposed in the paper, and what is its central design principle?"
        },
        {
            "expected_keywords": ["Transformer", "attention", "recurrence", "convolutions"],
            "ground_truth": "The Transformer is a sequence transduction architecture based entirely on attention mechanisms, dispensing with recurrence and convolutions."
        }
    ),
    (
        {
            "question": "What BLEU score did the Transformer big model achieve on the WMT 2014 English-to-German translation task?"
        },
        {
            "expected_keywords": ["28.4", "BLEU", "English-to-German"],
            "ground_truth": "The Transformer big model achieved 28.4 BLEU on the WMT 2014 English-to-German translation task."
        }
    ),
    (
        {
            "question": "What are the computational complexity per layer, number of sequential operations, and maximum path length for self-attention?"
        },
        {
            "expected_keywords": ["O(n^2", "O(1)", "maximum path length"],
            "ground_truth": "Self-attention has O(n^2 * d) computational complexity per layer, O(1) sequential operations, and O(1) maximum path length."
        }
    ),
    (
        {
            "question": "How many identical layers are used in the Transformer base model's encoder and decoder, and what is the model dimension?"
        },
        {
            "expected_keywords": ["6", "encoder", "decoder", "512"],
            "ground_truth": "Both the encoder and decoder contain six identical layers, and the model dimension d_model is 512."
        }
    ),
    (
        {
            "question": "What are the two sub-layers in each Transformer encoder layer?"
        },
        {
            "expected_keywords": ["multi-head self-attention", "feed-forward network"],
            "ground_truth": "Each encoder layer contains a multi-head self-attention sub-layer and a position-wise fully connected feed-forward network."
        }
    ),
    (
        {
            "question": "Why does scaled dot-product attention divide query-key dot products by the square root of the key dimension?"
        },
        {
            "expected_keywords": ["sqrt(dk)", "softmax", "small gradients", "large dot products"],
            "ground_truth": "When the key dimension is large, query-key dot products can become large in magnitude, pushing softmax into regions with extremely small gradients. Scaling the dot products by the square root of the key dimension counteracts this effect."
        }
    ),
    (
        {
            "question": "How many attention heads does the original Transformer use, and what are the key and value dimensions of each head?"
        },
        {
            "expected_keywords": ["8", "64", "attention heads"],
            "ground_truth": "The Transformer uses eight parallel attention heads, with key dimension dk = 64 and value dimension dv = 64 for each head."
        }
    ),
    (
        {
            "question": "How does multi-head attention differ from using a single attention head?"
        },
        {
            "expected_keywords": ["different representation subspaces", "parallel", "concatenated"],
            "ground_truth": "Multi-head attention applies learned linear projections to queries, keys, and values, performs attention in parallel, concatenates the resulting outputs, and projects them again. This allows the model to attend jointly to information from different representation subspaces and positions."
        }
    ),
    (
        {
            "question": "Why does the Transformer require positional encodings, and which functions does the original paper use?"
        },
        {
            "expected_keywords": ["positional encodings", "sine", "cosine", "sequence order"],
            "ground_truth": "Because the Transformer has neither recurrence nor convolution, positional encodings inject information about token positions. The original model uses sine and cosine functions of different frequencies and adds the encodings to the input embeddings."
        }
    ),
    (
        {
            "question": "How does masking in the Transformer decoder preserve autoregressive generation?"
        },
        {
            "expected_keywords": ["masking", "subsequent positions", "softmax", "autoregressive"],
            "ground_truth": "The decoder masks illegal connections by setting their attention logits to negative infinity before softmax. This prevents each position from attending to subsequent positions, so predictions depend only on previously known output tokens."
        }
    ),
    (
        {
            "question": "What is the computational complexity of a recurrent layer compared with a self-attention layer?"
        },
        {
            "expected_keywords": ["O(n * d^2)", "O(n^2 * d)", "recurrent", "self-attention"],
            "ground_truth": "A recurrent layer has O(n * d^2) computational complexity per layer, whereas a self-attention layer has O(n^2 * d) complexity. Self-attention also requires O(1) sequential operations compared with O(n) for recurrence."
        }
    ),
    (
        {
            "question": "What training dataset and hardware were used for the WMT 2014 English-to-German translation experiments?"
        },
        {
            "expected_keywords": ["4.5 million", "sentence pairs", "8 NVIDIA P100 GPUs", "byte-pair encoding"],
            "ground_truth": "The English-to-German experiments used approximately 4.5 million sentence pairs, byte-pair encoding with a shared source-target vocabulary of about 37,000 tokens, and one machine equipped with eight NVIDIA P100 GPUs."
        }
    ),
    (
        {
            "question": "How long was the Transformer base model trained, and how many training steps did it use?"
        },
        {
            "expected_keywords": ["100,000", "12 hours", "base model", "P100"],
            "ground_truth": "The Transformer base model was trained for 100,000 steps, taking approximately 12 hours on eight NVIDIA P100 GPUs."
        }
    ),
    (
        {
            "question": "What optimizer and learning-rate schedule did the Transformer paper use?"
        },
        {
            "expected_keywords": ["Adam", "warmup_steps", "4000", "inverse square root"],
            "ground_truth": "The Transformer used the Adam optimizer with beta1 = 0.9, beta2 = 0.98, and epsilon = 10^-9. Its learning rate increased linearly during the first 4,000 warmup steps and then decreased proportionally to the inverse square root of the step number."
        }
    ),
    (
        {
            "question": "What effect did label smoothing have on the Transformer during training?"
        },
        {
            "expected_keywords": ["0.1", "perplexity", "accuracy", "BLEU"],
            "ground_truth": "The Transformer used label smoothing with a value of 0.1. It increased perplexity because the model became less certain, but improved accuracy and BLEU score."
        }
    ),
    (
        {
            "question": "How did replacing sinusoidal positional encodings with learned positional embeddings affect the base model's performance?"
        },
        {
            "expected_keywords": ["learned positional embeddings", "sinusoidal", "25.7", "25.8"],
            "ground_truth": "Replacing sinusoidal positional encodings with learned positional embeddings produced nearly identical results. On the English-to-German development set, the reported BLEU scores were 25.7 and 25.8, respectively."
        }
    ),
    (
        {
            "question": "Why is self-attention potentially advantageous for learning long-range dependencies compared with recurrent layers?"
        },
        {
            "expected_keywords": ["constant number", "O(1)", "O(n)", "long-range dependencies"],
            "ground_truth": "Self-attention connects all positions with a constant maximum path length of O(1), while recurrent layers have a maximum path length of O(n). Shorter paths can make long-range dependencies easier to learn, and self-attention allows parallel computation across positions."
        }
    ),
     (
        {
            "question": "What is harness self-evolution, and what happens to the model weights during this process?"
        },
        {
            "expected_keywords": ["harness self-evolution", "feedback", "model weights", "fixed"],
            "ground_truth": "Harness self-evolution is a feedback-driven process that revises an agent's software scaffolding, such as prompts, memory, workflows, and executable components, while keeping the task model's weights fixed. Accepted revisions persist and influence subsequent runs."
        }
    ),
    (
        {
            "question": "What two requirements distinguish harness self-evolution from simply storing execution logs?"
        },
        {
            "expected_keywords": ["operational feedback", "update continuity", "persistence across runs"],
            "ground_truth": "Harness self-evolution requires operational feedback and continuity of the update mechanism, as well as persistence of accepted revisions across runs. The revisions must influence later executions, and the mechanism for further updates must remain available."
        }
    ),
    (
        {
            "question": "How does the paper distinguish interpreted-state updates from executable revisions?"
        },
        {
            "expected_keywords": ["interpreted state", "prompts", "memory", "executable revisions", "runtime control logic"],
            "ground_truth": "Interpreted-state updates change instructions, stored experience, or task-level coordination over predefined capabilities. Executable revisions change capability implementations, runtime control logic, or plugin-based service composition."
        }
    ),
    (
        {
            "question": "What are the seven runtime functions of an agent harness identified in Table 1?"
        },
        {
            "expected_keywords": ["agent loop", "tool orchestration", "context construction", "memory", "context compaction", "scheduling", "execution infrastructure"],
            "ground_truth": "The seven runtime functions are agent loop and execution control; tool and skill orchestration; context construction; memory and state management; context compaction; scheduling and coordination; and execution infrastructure."
        }
    ),
    (
        {
            "question": "What are the four main stages or aspects of harness updating described in the evolutionary perspective?"
        },
        {
            "expected_keywords": ["representation", "variation", "fitness", "selection"],
            "ground_truth": "The evolutionary perspective separates the representation of mutable state, the variation mechanism that generates edits, the assessment of candidate fitness, and the selection of state to retain."
        }
    ),
    (
        {
            "question": "What does the paper say about the relationship between empirical improvements and generalization?"
        },
        {
            "expected_keywords": ["empirical fitness", "overfit", "generalization", "retention"],
            "ground_truth": "Selecting revisions based on empirical fitness alone does not establish generalization or retention. Accepted revisions can overfit the evaluation tasks or cause previously acquired capabilities to be forgotten, motivating stability analysis, regression budgets, and independent validation."
        }
    ),
    (
        {
            "question": "Under what conditions does the paper derive an expected generalization bound for harness self-evolution?"
        },
        {
            "expected_keywords": ["continual PAC-learning", "algorithmic stability", "independent tasks", "fixed model", "fixed evaluator"],
            "ground_truth": "The analysis assumes tasks are drawn independently within and across rounds from an exogenous distribution, auxiliary randomness and external assessment samples are independent of those task draws, and the model and evaluator are fixed. Under these assumptions, algorithmic stability bounds the magnitude of the expected generalization gap."
        }
    ),
    (
        {
            "question": "How does the paper define forgetting, and how can accumulated regressions be bounded?"
        },
        {
            "expected_keywords": ["Fs,k", "historical risk", "regression budgets", "telescoping"],
            "ground_truth": "Forgetting from round s to round k is defined as Fs,k = Rs(Hk) - Rs(Hs), where positive values indicate forgetting and negative values indicate backward transfer. If each later transition satisfies its assumed historical-risk regression budget, accumulated forgetting is bounded by the sum of those budgets."
        }
    ),
    (
        {
            "question": "Why does the paper require fresh, independent validation data when admitting a candidate harness revision?"
        },
        {
            "expected_keywords": ["fresh validation", "independent tasks", "Hoeffding", "union bound", "overfitting"],
            "ground_truth": "Fresh, independent validation data helps control uncertainty when comparing candidate revisions with the incumbent. Under the paper's assumptions, a Hoeffding bound and a union bound account for the candidate comparisons across rounds. Reusing data that influenced candidate construction can invalidate the assumed independence and make admission overly optimistic."
        }
    ),
    (
        {
            "question": "What is the paper's high-probability admission rule for a candidate harness revision?"
        },
        {
            "expected_keywords": ["negative upper bounds", "delta", "K", "T", "m", "validation"],
            "ground_truth": "The paper accepts only candidates whose estimated risk difference plus the Hoeffding uncertainty term is negative. For at most K candidates per round over T rounds and m fresh independent validation tasks, the upper bound is the empirical paired risk difference plus sqrt(2 ln(KT/delta) / m), with probability at least 1 - delta under the stated assumptions."
        }
    ),
    (
        {
            "question": "Why does a successful admission test on the historical task mixture not guarantee safety on every future task distribution?"
        },
        {
            "expected_keywords": ["historical mixture", "future distributions", "groupwise retention", "distribution shift"],
            "ground_truth": "The admission certificate concerns risk on the historical mixture, not arbitrary future distributions. A mixture-level improvement does not guarantee retention for every task group or under distribution shift, so continued validation is necessary as tasks and evaluators change."
        }
    ),
    (
        {
            "question": "In the SpreadsheetBench comparison, what accuracy did GPT-5.4-nano achieve with HarnessOpt and skill, and how did it compare with GPT-5.5 using SkillOpt?"
        },
        {
            "expected_keywords": ["77.58%", "76.20%", "GPT-5.4-nano", "GPT-5.5", "HarnessOpt"],
            "ground_truth": "GPT-5.4-nano with HarnessOpt and skill achieved 77.58% accuracy, compared with 76.20% for GPT-5.5 using SkillOpt. The paper presents this as an example of model-harness pairing, not proof of superiority at matched cost."
        }
    ),
    (
        {
            "question": "In the reported spreadsheet plugin example, how did the first task's tool calls, token usage, and wall time change when using Everything Is a Plugin?"
        },
        {
            "expected_keywords": ["13", "3", "41,119", "47,572", "15.49", "24.00"],
            "ground_truth": "For the first task, tool calls decreased from 13 to 3. Token usage increased from 41,119 to 47,572, and wall time increased from 15.49 seconds to 24.00 seconds. Thus, the plugin reduced tool calls but increased first-task token usage and execution time."
        }
    ),
    (
        {
            "question": "After how many tasks did the reported cumulative Everything Is a Plugin comparison show savings of 110,351 tokens and 22.97 seconds?"
        },
        {
            "expected_keywords": ["five tasks", "110,351", "22.97", "cumulative savings"],
            "ground_truth": "The cumulative comparison reported savings of 110,351 tokens and 22.97 seconds after five tasks. These are reported as negative differences for Everything Is a Plugin minus the baseline."
        }
    ),
    (
        {
            "question": "What unresolved inconsistency does the paper identify in the reported first-task and cumulative plugin costs?"
        },
        {
            "expected_keywords": ["4,895", "6.74", "6,453", "8.51", "inconsistency"],
            "ground_truth": "The first-task totals imply an increase of 6,453 tokens and 8.51 seconds, whereas the cumulative table's first row reports increases of 4,895 tokens and 6.74 seconds. The paper identifies this as an unresolved difference and cautions against treating the two reports as one consistent cost trajectory."
        }
    ),
    (
        {
            "question": "What safeguards does the paper recommend for deploying self-evolving agents in robotics and physical environments?"
        },
        {
            "expected_keywords": ["emergency stops", "force limits", "speed limits", "authorization checks", "independent action checks"],
            "ground_truth": "The paper recommends keeping emergency stops, force and speed limits, and authorization checks outside the editable surface. It also recommends independent action checks, staged trials, and revision-linked command records because software rollback cannot reverse physical damage or injury."
        }
    ),
    (
        {
            "question": "Why does the paper argue that the final assessment function J must remain outside the editable harness?"
        },
        {
            "expected_keywords": ["final assessment", "independent measurement", "specification gaming", "evaluator"],
            "ground_truth": "The final assessment function J determines scoring, outcome verification, and the records that establish results. Keeping it outside the editable harness helps prevent a candidate from modifying the criteria used to judge itself. The paper also emphasizes independent measurement because optimizing against provisional or imperfect metrics can produce specification gaming."
        }
    ),
    (
        {
            "question": "What is the difference between retaining a candidate in an archive and adopting it into a running agent system?"
        },
        {
            "expected_keywords": ["archive membership", "adoption", "runtime", "subsequent tasks"],
            "ground_truth": "Archive membership preserves a candidate for further search. Adoption requires promoting the selected program into the runtime used by later tasks. An exported program continues the self-evolution lifecycle only if its deployment retains an update mechanism."
        }
    ),
    (
        {
            "question": "What are the main risks of sharing persistent harness revisions across enterprise teams?"
        },
        {
            "expected_keywords": ["shared faulty procedures", "correlated errors", "permissions", "version-linked deployment", "staged rollout"],
            "ground_truth": "A workaround that is safe for one team may conflict with another team's permissions or data assumptions. Shared faulty procedures can produce correlated errors across teams. The paper recommends identifying affected consumers, withdrawing problematic versions, restoring dependent workflows, keeping version-linked deployment records, and using staged rollouts."
        }
    ),
    (
        {
            "question": "What research directions does the paper identify for understanding model-harness co-design?"
        },
        {
            "expected_keywords": ["budget-matched", "model selection", "skill optimization", "runtime revision", "weight updates"],
            "ground_truth": "The paper proposes budget-matched comparisons of model selection, skill optimization, runtime revision, and model-weight updates, accounting for adaptation and subsequent execution costs. It also proposes testing frozen, transferred, and re-evolved harnesses across model checkpoints to distinguish transferable improvements from gains that require further harness evolution."
        }
    ),
]
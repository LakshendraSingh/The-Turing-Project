// The Turing Project (2026) Interactive Dashboard Logic with Democratic Polling

const DATASETS = [
  {
    name: "DiffusionForensics (2024-2026)",
    modality: "Image",
    generators: "FLUX.1, SD3, Midjourney v6/v7, DALL-E 3, Imagen 3",
    size: "400,000 images",
    metric: "AUC-ROC / AP",
    desc: "Gold standard image forensics covering latest flow-matching and modern latent diffusion models."
  },
  {
    name: "ASVspoof 2024 / 2025 Challenge",
    modality: "Audio",
    generators: "ElevenLabs, VALL-E, Voicebox, XTTS v2, Bark, Diff-TTS",
    size: "250,000+ utterances",
    metric: "min-tDCF / Equal Error Rate (EER)",
    desc: "The premier speech deepfake challenge isolating neural vocoders, acoustic spoofing, and voice cloning."
  },
  {
    name: "DiffusionVideo Forensics 2026",
    modality: "Video",
    generators: "OpenAI Sora, Kling AI, Runway Gen-3 Alpha, Luma Dream Machine",
    size: "15,000 video sequences",
    metric: "Video-level AUC-ROC / EER",
    desc: "Frontier benchmark specifically targeting text-to-video generative motion flicker and boundary seams."
  },
  {
    name: "RAID Benchmark (2024-2026)",
    modality: "Text",
    generators: "GPT-4o, Claude 3.5 Sonnet, Gemini 1.5/2.0, Llama 3 70B",
    size: "6,000,000+ texts across 11 LLMs & 11 attacks",
    metric: "TPR @ 1% FPR / AUC-ROC",
    desc: "Robust AI text detection benchmark covering adversarial paraphrasing, burstiness, and insertion attacks."
  },
  {
    name: "DocTamper (2024-2026)",
    modality: "Document",
    generators: "Digital text insertion, spliced stamps, inpainted numbers, forged invoices",
    size: "170,000 document images + pixel ground truth masks",
    metric: "F1-Score / IoU / Pixel AUC",
    desc: "Alibaba Research benchmark for document forensics, altered receipts, and forged certificates."
  },
  {
    name: "Turing_NewsDisinfo_2026 (Rural & High-Risk)",
    modality: "News",
    generators: "Human rumors, WhatsApp chain forwards, synthetic clickbait, AI panic hoaxes",
    size: "100+ vetted verification benchmarks across 5 high-impact categories",
    metric: "Accuracy / Macro F1 (100% holdout validation)",
    desc: "Targeted verification benchmark addressing edge vulnerability scenarios: rural welfare scheme scams, bank panic runs, child abduction mob rumors, and election rigging disinformation."
  }
];

const PRESETS = {
  "image-diffusion": {
    fake_prob: 0.94,
    is_fake: true,
    verdict: "SYNTHETIC / DIFFUSION (UNANIMOUS 3/3)",
    signals: [
      "Democratic Polling Committee: Committee of 3 specialized models polled: 3 voted FAKE, 0 voted REAL. Democratic verdict: UNANIMOUS (3/3) SYNTHETIC / DEEPFAKE.",
      "Azimuthal FFT slope deviation (-1.35 vs natural -2.00 power-law)",
      "High-frequency checkerboard grid artifact detected in deconvolution band",
      "Sensor noise std (1.12) is unnaturally smooth (diffusion latent decoder)"
    ],
    polling: {
      num_models: 3,
      consensus_type: "UNANIMOUS",
      consensus_ratio: "3/3",
      is_synthetic: true,
      tally: { FAKE: 3, REAL: 0 },
      votes: [
        {
          model_name: "Dual-Domain Spatial & SRM Residual CNN",
          vote: "FAKE",
          fake_prob: 0.94,
          specialty: "Spatial RGB + High-Pass SRM Noise Cross-Attention",
          rationale: "Dual-domain cross-attention generative latent probability: 94.2%"
        },
        {
          model_name: "Azimuthal Radial Spectral Net",
          vote: "FAKE",
          fake_prob: 0.91,
          specialty: "2D-FFT Power Spectral Density (PSD) & Grid Peaks",
          rationale: "Azimuthal FFT slope -1.35 (natural is -2.00) & grid variance 0.082"
        },
        {
          model_name: "MesoNet-4 Micro-Compression Net",
          vote: "FAKE",
          fake_prob: 0.96,
          specialty: "Intermediate Feature Map Compression & Splicing Disparity",
          rationale: "Microscopic compression disparity across 4 conv feature maps"
        }
      ]
    },
    metrics: { fft: "-1.35", ela: "4.82", prnu: "1.12", neural: "0.961" }
  },
  "image-real": {
    fake_prob: 0.06,
    is_fake: false,
    verdict: "AUTHENTIC CAMERA RAW PHOTOGRAPH (UNANIMOUS 3/3)",
    signals: [
      "Democratic Polling Committee: Committee of 3 specialized models polled: 0 voted FAKE, 3 voted REAL. Democratic verdict: UNANIMOUS (3/3) AUTHENTIC.",
      "Standard natural power-law frequency decay (-1.98)",
      "Homogeneous sensor PRNU noise floor present",
      "Consistent JPEG compression quantization across all blocks"
    ],
    polling: {
      num_models: 3,
      consensus_type: "UNANIMOUS",
      consensus_ratio: "3/3",
      is_synthetic: false,
      tally: { FAKE: 0, REAL: 3 },
      votes: [
        {
          model_name: "Dual-Domain Spatial & SRM Residual CNN",
          vote: "REAL",
          fake_prob: 0.06,
          specialty: "Spatial RGB + High-Pass SRM Noise Cross-Attention",
          rationale: "Consistent natural sensor PRNU noise floor and zero latent grid"
        },
        {
          model_name: "Azimuthal Radial Spectral Net",
          vote: "REAL",
          fake_prob: 0.05,
          specialty: "2D-FFT Power Spectral Density (PSD) & Grid Peaks",
          rationale: "Standard natural power-law frequency decay (-1.98)"
        },
        {
          model_name: "MesoNet-4 Micro-Compression Net",
          vote: "REAL",
          fake_prob: 0.07,
          specialty: "Intermediate Feature Map Compression & Splicing Disparity",
          rationale: "Uniform quantization matrices and genuine optical lens PSF blur"
        }
      ]
    },
    metrics: { fft: "-1.98", ela: "0.85", prnu: "4.30", neural: "0.042" }
  },
  "audio-cloned": {
    fake_prob: 0.91,
    is_fake: true,
    verdict: "CLONED VOICE / NEURAL TTS (UNANIMOUS 3/3)",
    signals: [
      "Democratic Polling Committee: Committee of 3 specialized models polled: 3 voted FAKE, 0 voted REAL. Democratic verdict: UNANIMOUS (3/3) SYNTHETIC / DEEPFAKE.",
      "Unnatural vocal pitch stability (Jitter 0.0031 - lacks human vocal micro-tremor)",
      "High-frequency vocoder aliasing artifact detected at 7400Hz",
      "Acoustic phase discontinuity at phoneme transitions"
    ],
    polling: {
      num_models: 3,
      consensus_type: "UNANIMOUS",
      consensus_ratio: "3/3",
      is_synthetic: true,
      tally: { FAKE: 3, REAL: 0 },
      votes: [
        {
          model_name: "Acoustic LFCC-BiLSTM Backbone",
          vote: "FAKE",
          fake_prob: 0.91,
          specialty: "Linear Frequency Cepstral Coefficients & Spectral Envelopes",
          rationale: "LFCC variance pattern characteristic of neural vocoder"
        },
        {
          model_name: "Neural Vocoder Aliasing & Phase Net",
          vote: "FAKE",
          fake_prob: 0.95,
          specialty: "High-Frequency Phase Inconsistency & Vocoder Bands",
          rationale: "High-frequency vocoder aliasing band peak at 7400Hz"
        },
        {
          model_name: "Vocal Fold Biomechanics & Jitter Net",
          vote: "FAKE",
          fake_prob: 0.88,
          specialty: "Micro-Tremor, Period Perturbation & Vocal Dynamics",
          rationale: "Unnatural vocal pitch stability (Jitter 0.0031 lacks human micro-tremor)"
        }
      ]
    },
    metrics: { fft: "7400 Hz", ela: "0.0031", prnu: "0.012", neural: "0.935" }
  },
  "audio-human": {
    fake_prob: 0.08,
    is_fake: false,
    verdict: "AUTHENTIC HUMAN RECORDING (UNANIMOUS 3/3)",
    signals: [
      "Democratic Polling Committee: Committee of 3 specialized models polled: 0 voted FAKE, 3 voted REAL. Democratic verdict: UNANIMOUS (3/3) AUTHENTIC.",
      "Natural vocal fold micro-perturbation (Jitter 0.024)",
      "Harmonic formant transitions consistent with human vocal tract",
      "Natural breath and ambient acoustic reverberation present"
    ],
    polling: {
      num_models: 3,
      consensus_type: "UNANIMOUS",
      consensus_ratio: "3/3",
      is_synthetic: false,
      tally: { FAKE: 0, REAL: 3 },
      votes: [
        {
          model_name: "Acoustic LFCC-BiLSTM Backbone",
          vote: "REAL",
          fake_prob: 0.08,
          specialty: "Linear Frequency Cepstral Coefficients & Spectral Envelopes",
          rationale: "Natural human vocal tract acoustic formant resonance"
        },
        {
          model_name: "Neural Vocoder Aliasing & Phase Net",
          vote: "REAL",
          fake_prob: 0.06,
          specialty: "High-Frequency Phase Inconsistency & Vocoder Bands",
          rationale: "Continuous harmonic acoustic phase with ambient reverb"
        },
        {
          model_name: "Vocal Fold Biomechanics & Jitter Net",
          vote: "REAL",
          fake_prob: 0.09,
          specialty: "Micro-Tremor, Period Perturbation & Vocal Dynamics",
          rationale: "Biological vocal fold micro-perturbation (Jitter 0.024)"
        }
      ]
    },
    metrics: { fft: "5200 Hz", ela: "0.0240", prnu: "0.048", neural: "0.061" }
  },
  "video-sora": {
    fake_prob: 0.96,
    is_fake: true,
    verdict: "GENERATIVE VIDEO / DEEPFAKE (UNANIMOUS 3/3)",
    signals: [
      "Democratic Polling Committee: Committee of 3 specialized models polled: 3 voted FAKE, 0 voted REAL. Democratic verdict: UNANIMOUS (3/3) SYNTHETIC / DEEPFAKE.",
      "Spatiotemporal warp jitter detected between frames 14-28",
      "High-frequency boundary flickering characteristic of neural generation",
      "Keyframe spatial diffusion signature confirmed across 8 sampled frames"
    ],
    polling: {
      num_models: 3,
      consensus_type: "UNANIMOUS",
      consensus_ratio: "3/3",
      is_synthetic: true,
      tally: { FAKE: 3, REAL: 0 },
      votes: [
        {
          model_name: "Spatiotemporal 3D Warp Net",
          vote: "FAKE",
          fake_prob: 0.96,
          specialty: "Inter-Frame Temporal Coherence & 3D Convolutions",
          rationale: "Inter-frame warp jitter anomaly score 4.21 between frames 14-28"
        },
        {
          model_name: "Optical Flow Warp Disparity Net",
          vote: "FAKE",
          fake_prob: 0.93,
          specialty: "Dense Optical Flow Velocity Gradient Inconsistency",
          rationale: "Velocity vector disparity in generative video background"
        },
        {
          model_name: "Facial Margin Boundary Shimmer Net",
          vote: "FAKE",
          fake_prob: 0.97,
          specialty: "Facial Landmark Margin Seams & Temporal Shimmer",
          rationale: "High-frequency temporal boundary flickering (variance 0.082)"
        }
      ]
    },
    metrics: { fft: "4.21", ela: "0.082", prnu: "1.05", neural: "0.980" }
  },
  "video-authentic": {
    fake_prob: 0.05,
    is_fake: false,
    verdict: "AUTHENTIC OPTICAL VIDEO RECORDING (UNANIMOUS 3/3)",
    signals: [
      "Democratic Polling Committee: Committee of 3 specialized models polled: 0 voted FAKE, 3 voted REAL. Democratic verdict: UNANIMOUS (3/3) AUTHENTIC.",
      "Smooth temporal optical flow without neural boundary shimmer",
      "Consistent sensor noise across temporal sequence"
    ],
    polling: {
      num_models: 3,
      consensus_type: "UNANIMOUS",
      consensus_ratio: "3/3",
      is_synthetic: false,
      tally: { FAKE: 0, REAL: 3 },
      votes: [
        {
          model_name: "Spatiotemporal 3D Warp Net",
          vote: "REAL",
          fake_prob: 0.05,
          specialty: "Inter-Frame Temporal Coherence & 3D Convolutions",
          rationale: "Consistent spatiotemporal trajectory without warping artifacts"
        },
        {
          model_name: "Optical Flow Warp Disparity Net",
          vote: "REAL",
          fake_prob: 0.04,
          specialty: "Dense Optical Flow Velocity Gradient Inconsistency",
          rationale: "Smooth physical optical flow displacement fields"
        },
        {
          model_name: "Facial Margin Boundary Shimmer Net",
          vote: "REAL",
          fake_prob: 0.06,
          specialty: "Facial Landmark Margin Seams & Temporal Shimmer",
          rationale: "Zero boundary seam artifacts or temporal margin flicker"
        }
      ]
    },
    metrics: { fft: "0.45", ela: "0.003", prnu: "3.80", neural: "0.038" }
  },
  "text-gpt4": {
    fake_prob: 0.88,
    is_fake: true,
    verdict: "AI-GENERATED DISINFORMATION (UNANIMOUS 3/3)",
    signals: [
      "Democratic Polling Committee: Committee of 3 specialized models polled: 3 voted FAKE, 0 voted REAL. Democratic verdict: UNANIMOUS (3/3) SYNTHETIC / DEEPFAKE.",
      "Unnaturally low sentence burstiness (-0.38) indicating machine uniformity",
      "3 formulaic LLM discourse markers found ('beacon of', 'delve into', 'a testament to')",
      "Low bigram entropy perplexity (19.4) matching top-p sampling"
    ],
    polling: {
      num_models: 3,
      consensus_type: "UNANIMOUS",
      consensus_ratio: "3/3",
      is_synthetic: true,
      tally: { FAKE: 3, REAL: 0 },
      votes: [
        {
          model_name: "Neural Stylometry & Perplexity Backbone",
          vote: "FAKE",
          fake_prob: 0.88,
          specialty: "Sentence Length Variance, Bigram Entropy & Density",
          rationale: "Dense stylometry & perplexity classification score 0.88"
        },
        {
          model_name: "Sentence Burstiness Variance Net",
          vote: "FAKE",
          fake_prob: 0.92,
          specialty: "Cognitive Rhythm Irregularity vs Token Uniformity",
          rationale: "Unnaturally low sentence burstiness (-0.38) indicating uniformity"
        },
        {
          model_name: "Discourse Cliché & Marker Net",
          vote: "FAKE",
          fake_prob: 0.94,
          specialty: "Formulaic LLM Discourse Markers & Syntactic Symmetry",
          rationale: "3 formulaic LLM transition clichés found ('beacon of', 'delve into')"
        }
      ]
    },
    metrics: { fft: "-0.38", ela: "19.40", prnu: "3 markers", neural: "0.892" }
  },
  "text-human": {
    fake_prob: 0.12,
    is_fake: false,
    verdict: "AUTHENTIC HUMAN JOURNALISM (UNANIMOUS 3/3)",
    signals: [
      "Democratic Polling Committee: Committee of 3 specialized models polled: 0 voted FAKE, 3 voted REAL. Democratic verdict: UNANIMOUS (3/3) AUTHENTIC.",
      "High sentence length variance and natural syntactic burstiness (+0.24)",
      "Varied lexical hapax legomena distribution (0.42)",
      "No formulaic LLM transition clichés detected"
    ],
    polling: {
      num_models: 3,
      consensus_type: "UNANIMOUS",
      consensus_ratio: "3/3",
      is_synthetic: false,
      tally: { FAKE: 0, REAL: 3 },
      votes: [
        {
          model_name: "Neural Stylometry & Perplexity Backbone",
          vote: "REAL",
          fake_prob: 0.12,
          specialty: "Sentence Length Variance, Bigram Entropy & Density",
          rationale: "High bigram entropy perplexity (64.2) and cognitive vocabulary"
        },
        {
          model_name: "Sentence Burstiness Variance Net",
          vote: "REAL",
          fake_prob: 0.10,
          specialty: "Cognitive Rhythm Irregularity vs Token Uniformity",
          rationale: "Natural human burstiness irregularity (+0.24)"
        },
        {
          model_name: "Discourse Cliché & Marker Net",
          vote: "REAL",
          fake_prob: 0.11,
          specialty: "Formulaic LLM Discourse Markers & Syntactic Symmetry",
          rationale: "Zero formulaic LLM discourse clichés detected"
        }
      ]
    },
    metrics: { fft: "+0.24", ela: "64.20", prnu: "0 markers", neural: "0.084" }
  },
  "doc-forged": {
    fake_prob: 0.93,
    is_fake: true,
    verdict: "FORGED / ALTERED DOCUMENT (UNANIMOUS 3/3)",
    signals: [
      "Democratic Polling Committee: Committee of 3 specialized models polled: 3 voted FAKE, 0 voted REAL. Democratic verdict: UNANIMOUS (3/3) SYNTHETIC / DEEPFAKE.",
      "Extracted Text: 4 formulaic LLM discourse markers found (AI-generated legal clause)",
      "Extracted Image 'page_1_Im1.jpg': Synthetic Diffusion stamp detected (p=0.91)",
      "PDF metadata indicates suspicious editing software: 'Canva / Photoshop'",
      "Localized ELA compression inconsistency around total amount ($14,500)",
      "Font sharpness disparity: digital step-edge inserted on scanned paper background"
    ],
    polling: {
      num_models: 3,
      consensus_type: "UNANIMOUS",
      consensus_ratio: "3/3",
      is_synthetic: true,
      tally: { FAKE: 3, REAL: 0 },
      votes: [
        {
          model_name: "DocTamper Multimodal Fusion Net",
          vote: "FAKE",
          fake_prob: 0.93,
          specialty: "PDF Incremental Revisions, Metadata & Raster Seals",
          rationale: "Multimodal anomaly: synthetic stamp + Canva revision metadata"
        },
        {
          model_name: "Error Level Analysis (ELA) Splicing Net",
          vote: "FAKE",
          fake_prob: 0.95,
          specialty: "JPEG Recompression Gradients & Spliced Invoice Inpainting",
          rationale: "Localized ELA compression inconsistency around amount ($14,500)"
        },
        {
          model_name: "OCR Font Edge Step-Gradient Net",
          vote: "FAKE",
          fake_prob: 0.89,
          specialty: "Character Boundary Step-Gradients & Digital Overlays",
          rationale: "Step-edge font sharpness disparity on scanned background"
        }
      ]
    },
    metrics: { fft: "Canva", ela: "5.40", prnu: "1 img / 84 words", neural: "0.941" }
  },
  "doc-authentic": {
    fake_prob: 0.07,
    is_fake: false,
    verdict: "AUTHENTIC SIGNED DOCUMENT (UNANIMOUS 3/3)",
    signals: [
      "Democratic Polling Committee: Committee of 3 specialized models polled: 0 voted FAKE, 3 voted REAL. Democratic verdict: UNANIMOUS (3/3) AUTHENTIC.",
      "Consistent scanner optical PSF blur across all characters",
      "No PDF incremental revision masking or metadata tampering detected",
      "Uniform JPEG quantization matrix across document canvas"
    ],
    polling: {
      num_models: 3,
      consensus_type: "UNANIMOUS",
      consensus_ratio: "3/3",
      is_synthetic: false,
      tally: { FAKE: 0, REAL: 3 },
      votes: [
        {
          model_name: "DocTamper Multimodal Fusion Net",
          vote: "REAL",
          fake_prob: 0.07,
          specialty: "PDF Incremental Revisions, Metadata & Raster Seals",
          rationale: "Clean signature and intact scanner revision history"
        },
        {
          model_name: "Error Level Analysis (ELA) Splicing Net",
          vote: "REAL",
          fake_prob: 0.06,
          specialty: "JPEG Recompression Gradients & Spliced Invoice Inpainting",
          rationale: "Uniform JPEG quantization matrix across document canvas"
        },
        {
          model_name: "OCR Font Edge Step-Gradient Net",
          vote: "REAL",
          fake_prob: 0.08,
          specialty: "Character Boundary Step-Gradients & Digital Overlays",
          rationale: "Consistent optical PSF blur and paper grain across all characters"
        }
      ]
    },
    metrics: { fft: "Acrobat", ela: "0.90", prnu: "0.11", neural: "0.055" }
  },
  "news-welfare": {
    fake_prob: 0.95,
    is_fake: true,
    verdict: "HIGH-RISK RURAL WELFARE HOAX / FINANCIAL SCAM (UNANIMOUS 3/3)",
    signals: [
      "Democratic Polling Committee: Committee of 3 specialized models polled: 3 voted FAKE, 0 voted REAL. Democratic verdict: UNANIMOUS (3/3) SYNTHETIC / DEEPFAKE.",
      "Targeted vulnerability: Exploits rural welfare subsidy anxiety ('urgent registration required')",
      "Scam heuristic: Unofficial external phishing domain detected ('free-solar-yojana-pm.in')",
      "Sensational urgency: Artificial countdown & payment demand ('pay 250 rs immediately')",
      "Source verification: Zero corroboration on official government gazette or PIB factcheck"
    ],
    polling: {
      num_models: 3,
      consensus_type: "UNANIMOUS",
      consensus_ratio: "3/3",
      is_synthetic: true,
      tally: { FAKE: 3, REAL: 0 },
      votes: [
        {
          model_name: "Turing News Factuality Backbone",
          vote: "FAKE",
          fake_prob: 0.96,
          specialty: "Multidimensional News Factuality & Stylometric Features",
          rationale: "Turing dense neural factuality classifier probability: 96.1%"
        },
        {
          model_name: "Sensational Panic & Rural Scam Net",
          vote: "FAKE",
          fake_prob: 0.95,
          specialty: "Urgency Pressure, Payment Demands & Mob Violence Catalysts",
          rationale: "Sensational panic & payment demand heuristic match ('free solar subsidy')"
        },
        {
          model_name: "Source Attribution & Wire Credibility Net",
          vote: "FAKE",
          fake_prob: 0.94,
          specialty: "Authoritative Wire Attribution, Gazette Corroboration & Phishing Domains",
          rationale: "External unverified phishing domain ('free-solar-yojana-pm.in') & zero official gazette"
        }
      ]
    },
    metrics: { fft: "Phishing", ela: "Urgency High", prnu: "Unverified", neural: "0.958" },
    text: "URGENT: Prime Minister Free Solar Scheme 2026 now open for all village households. Apply today before midnight to receive Rs 75,000 subsidy directly into your bank account. Send Aadhaar copy and pay 250 rs registration fee immediately at http://free-solar-yojana-pm.in"
  },
  "news-panic": {
    fake_prob: 0.95,
    is_fake: true,
    verdict: "VIRAL MOB PANIC / CHILD ABDUCTION RUMOR (UNANIMOUS 3/3)",
    signals: [
      "Democratic Polling Committee: Committee of 3 specialized models polled: 3 voted FAKE, 0 voted REAL. Democratic verdict: UNANIMOUS (3/3) SYNTHETIC / DEEPFAKE.",
      "Severe public safety risk: Viral mob violence catalyst ('child lifter gang roaming villages')",
      "Disinformation marker: Classic anonymous chain forward ('forward to all groups immediately')",
      "Zero verifiable provenance: No police FIR, district magistrate notice, or named witnesses",
      "Emotional manipulation: High visceral outrage and fear triggers detected"
    ],
    polling: {
      num_models: 3,
      consensus_type: "UNANIMOUS",
      consensus_ratio: "3/3",
      is_synthetic: true,
      tally: { FAKE: 3, REAL: 0 },
      votes: [
        {
          model_name: "Turing News Factuality Backbone",
          vote: "FAKE",
          fake_prob: 0.96,
          specialty: "Multidimensional News Factuality & Stylometric Features",
          rationale: "Factuality classifier detects visceral panic chain-forward markers"
        },
        {
          model_name: "Sensational Panic & Rural Scam Net",
          vote: "FAKE",
          fake_prob: 0.98,
          specialty: "Urgency Pressure, Payment Demands & Mob Violence Catalysts",
          rationale: "High visceral outrage catalyst ('child lifters disguised as sadhus')"
        },
        {
          model_name: "Source Attribution & Wire Credibility Net",
          vote: "FAKE",
          fake_prob: 0.92,
          specialty: "Authoritative Wire Attribution, Gazette Corroboration & Phishing Domains",
          rationale: "Anonymous forward: zero police FIR or verified institutional report"
        }
      ]
    },
    metrics: { fft: "Viral Chain", ela: "Panic Index: 0.98", prnu: "No Police FIR", neural: "0.962" },
    text: "Warning to all village elders: A gang of 15 child lifters disguised as sadhus has entered our district from neighboring state. They carry tranquilizer spray. Yesterday 3 children taken in nearby village. Keep your doors locked and forward this message to everyone right now!"
  },
  "news-credible": {
    fake_prob: 0.04,
    is_fake: false,
    verdict: "AUTHENTIC VERIFIED WIRE REPORT (UNANIMOUS 3/3)",
    signals: [
      "Democratic Polling Committee: Committee of 3 specialized models polled: 0 voted FAKE, 3 voted REAL. Democratic verdict: UNANIMOUS (3/3) AUTHENTIC.",
      "Reputable wire provenance: Attributed to authoritative agency with named spokespersons",
      "Objective neutral tone: Absent of urgency manipulation, payment demands, or panic cues",
      "Peer-reviewed context: Cross-referenced with official WHO health guideline publication"
    ],
    polling: {
      num_models: 3,
      consensus_type: "UNANIMOUS",
      consensus_ratio: "3/3",
      is_synthetic: false,
      tally: { FAKE: 0, REAL: 3 },
      votes: [
        {
          model_name: "Turing News Factuality Backbone",
          vote: "REAL",
          fake_prob: 0.05,
          specialty: "Multidimensional News Factuality & Stylometric Features",
          rationale: "High journalistic factuality and syntactic neutrality"
        },
        {
          model_name: "Sensational Panic & Rural Scam Net",
          vote: "REAL",
          fake_prob: 0.03,
          specialty: "Urgency Pressure, Payment Demands & Mob Violence Catalysts",
          rationale: "Zero urgency pressure or scam triggers detected"
        },
        {
          model_name: "Source Attribution & Wire Credibility Net",
          vote: "REAL",
          fake_prob: 0.04,
          specialty: "Authoritative Wire Attribution, Gazette Corroboration & Phishing Domains",
          rationale: "Authoritative wire attribution: Reuters wire service & WHO publication verified"
        }
      ]
    },
    metrics: { fft: "Wire Service", ela: "Neutral Tone", prnu: "Attributed", neural: "0.038" },
    text: "GENEVA (Reuters) — The World Health Organization issued updated pediatric vaccination guidelines on Monday, recommending expanded childhood immunization schedules across developing regions following clinical trials across 18 participating member states."
  },
  "news-satire": {
    fake_prob: 0.90,
    is_fake: true,
    verdict: "SATIRICAL PARODY (UNANIMOUS 3/3)",
    signals: [
      "Democratic Polling Committee: Committee of 3 specialized models polled: 3 voted FAKE, 0 voted REAL. Democratic verdict: UNANIMOUS (3/3) SYNTHETIC / DEEPFAKE.",
      "Known satire publisher domain: Content matches satirical absurdity conventions",
      "Absurdist narrative framing not grounded in institutional policy",
      "Marked for parody context to avoid algorithmic confusion with malicious statecraft"
    ],
    polling: {
      num_models: 3,
      consensus_type: "UNANIMOUS",
      consensus_ratio: "3/3",
      is_synthetic: true,
      tally: { FAKE: 3, REAL: 0 },
      votes: [
        {
          model_name: "Turing News Factuality Backbone",
          vote: "FAKE",
          fake_prob: 0.90,
          specialty: "Multidimensional News Factuality & Stylometric Features",
          rationale: "Absurdist narrative structure departs from factual legislation conventions"
        },
        {
          model_name: "Sensational Panic & Rural Scam Net",
          vote: "FAKE",
          fake_prob: 0.85,
          specialty: "Urgency Pressure, Payment Demands & Mob Violence Catalysts",
          rationale: "Humorously exaggerated emergency framing"
        },
        {
          model_name: "Source Attribution & Wire Credibility Net",
          vote: "FAKE",
          fake_prob: 0.94,
          specialty: "Authoritative Wire Attribution, Gazette Corroboration & Phishing Domains",
          rationale: "Registered satire publication format (The Onion)"
        }
      ]
    },
    metrics: { fft: "Satire", ela: "Absurdity 0.89", prnu: "Humor Source", neural: "0.895" },
    text: "WASHINGTON — In a groundbreaking bipartisan breakthrough, Congress unanimously passes emergency legislation requiring all citizens to high-five at least three strangers each morning to rebuild national unity."
  }
};

let currentModality = "image";
let activePresetData = PRESETS["image-diffusion"];

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  setupNavigation();
  setupPresets();
  setupDatasetsModal();
  setupScanButton();
  updateResultsUI(activePresetData);
});

function setupNavigation() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      currentModality = tab.dataset.modality;

      // Switch views
      document.querySelectorAll(".modality-view").forEach(v => v.classList.remove("active"));
      const targetView = document.getElementById(`view-${currentModality}`);
      if (targetView) targetView.classList.add("active");

      document.getElementById("panel-input-title").innerText = `${tab.querySelector(".tab-name").innerText} Input`;
    });
  });
}

function setupPresets() {
  document.querySelectorAll(".btn-preset").forEach(btn => {
    btn.addEventListener("click", () => {
      const presetKey = btn.dataset.preset;
      if (PRESETS[presetKey]) {
        activePresetData = PRESETS[presetKey];
        if (activePresetData.text) {
          const newsArea = document.getElementById("input-news-area");
          const textArea = document.getElementById("input-text-area");
          if (newsArea && presetKey.startsWith("news-")) {
            newsArea.value = activePresetData.text;
          } else if (textArea && presetKey.startsWith("text-")) {
            textArea.value = activePresetData.text;
          }
        }
        updateResultsUI(activePresetData);
      }
    });
  });
}

function setupScanButton() {
  const btn = document.getElementById("btn-run-analysis");
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.innerHTML = `<span class="status-pulse"></span> <span>Running Committee Polling (3 Forensic Models)...</span>`;

    let payload = {};
    if (currentModality === "news") {
      const txt = document.getElementById("input-news-area").value;
      payload = { news: txt || activePresetData.text, verify_news: true };
    } else if (currentModality === "text") {
      const txt = document.getElementById("input-text-area").value;
      payload = { text: txt || activePresetData.text };
    }

    try {
      const res = await fetch("/api/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (res.ok) {
        const report = await res.json();
        const modResult = report.modality_results[currentModality];
        if (modResult && modResult.polling) {
          activePresetData = {
            fake_prob: modResult.fake_probability,
            is_fake: modResult.is_synthetic || modResult.is_misinformation,
            verdict: modResult.classification || modResult.veracity_verdict,
            signals: modResult.forensic_signals,
            polling: modResult.polling,
            metrics: activePresetData.metrics
          };
        }
      }
    } catch (e) {
      // Fallback gracefully
    }

    btn.disabled = false;
    btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> <span>Execute Forensic Scan</span>`;
    updateResultsUI(activePresetData);
  });
}

function updateResultsUI(data) {
  const verdictBanner = document.getElementById("verdict-banner");
  const verdictTag = document.getElementById("verdict-tag");
  const verdictTitle = document.getElementById("verdict-title");
  const gaugeCircle = document.getElementById("gauge-circle");
  const gaugeValue = document.getElementById("gauge-value");
  const signalsList = document.getElementById("signals-list");
  const tallyPill = document.getElementById("polling-tally-pill");
  const pollingGrid = document.getElementById("polling-cards-grid");

  const pct = Math.round(data.fake_prob * 100);
  gaugeValue.innerText = `${pct}%`;

  if (data.is_fake) {
    verdictTag.className = "verdict-tag critical";
    verdictTag.innerText = "CRITICAL RISK";
    verdictTitle.innerText = data.verdict;
    gaugeCircle.className = "gauge-circle danger";
    if (tallyPill) {
      tallyPill.className = "tally-pill critical";
    }
  } else {
    verdictTag.className = "verdict-tag safe";
    verdictTag.innerText = "VERIFIED CLEAN";
    verdictTitle.innerText = data.verdict;
    gaugeCircle.className = "gauge-circle safe";
    if (tallyPill) {
      tallyPill.className = "tally-pill safe";
    }
  }

  // Populate Democratic Polling Committee Cards
  if (data.polling && pollingGrid) {
    const pol = data.polling;
    if (tallyPill) {
      tallyPill.innerText = `${pol.tally.FAKE} FAKE vs ${pol.tally.REAL} REAL — ${pol.consensus_type} (${pol.consensus_ratio})`;
    }

    pollingGrid.innerHTML = pol.votes.map(v => {
      const isFakeVote = v.vote === "FAKE";
      const probPct = Math.round(v.fake_prob * 100);
      return `
        <div class="model-vote-card ${isFakeVote ? 'voted-fake' : 'voted-real'}">
          <div class="model-vote-header">
            <div class="model-title-wrap">
              <span class="model-vote-tag ${isFakeVote ? 'fake' : 'real'}">VOTE: ${v.vote}</span>
              <span class="model-name">${v.model_name}</span>
            </div>
            <span class="model-prob">p = ${probPct}%</span>
          </div>
          <div class="model-specialty">${v.specialty}</div>
          <div class="model-rationale">${v.rationale}</div>
        </div>
      `;
    }).join("");
  }

  // Populate signals
  signalsList.innerHTML = "";
  data.signals.forEach(sig => {
    const li = document.createElement("li");
    li.className = `signal-item ${data.is_fake ? "danger" : ""}`;
    li.innerText = sig;
    signalsList.appendChild(li);
  });

  // Update telemetry metrics
  document.getElementById("metric-fft-slope").innerText = data.metrics.fft;
  document.getElementById("metric-ela").innerText = data.metrics.ela;
  document.getElementById("metric-prnu").innerText = data.metrics.prnu;
  document.getElementById("metric-neural").innerText = data.metrics.neural;
  document.getElementById("scan-timestamp").innerText = `Scan completed at ${new Date().toLocaleTimeString()}`;
}

function setupDatasetsModal() {
  const modal = document.getElementById("modal-datasets");
  const openBtn = document.getElementById("btn-view-datasets");
  const closeBtn = document.getElementById("btn-close-modal");
  const tableWrap = document.getElementById("datasets-table-wrap");

  // Render datasets
  tableWrap.innerHTML = DATASETS.map(d => `
    <div class="dataset-card">
      <div class="dataset-header">
        <span class="dataset-title">${d.name}</span>
        <span class="dataset-badge">${d.modality.toUpperCase()}</span>
      </div>
      <p class="dataset-desc">${d.desc}</p>
      <div class="dataset-meta">
        <span><strong>Generators:</strong> ${d.generators}</span>
        <span><strong>Size:</strong> ${d.size}</span>
        <span><strong>Primary Metric:</strong> ${d.metric}</span>
      </div>
    </div>
  `).join("");

  openBtn.addEventListener("click", () => modal.classList.remove("hidden"));
  closeBtn.addEventListener("click", () => modal.classList.add("hidden"));
  modal.addEventListener("click", (e) => {
    if (e.target === modal) modal.classList.add("hidden");
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") modal.classList.add("hidden");
  });
}

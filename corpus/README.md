# Titan Fitness Policy Corpus (Fictional)

This is a fully fictional corpus of gym policy documents for **Titan Fitness**, a made-up gym chain, created for use as sample data in a RAG (Retrieval-Augmented Generation) project. None of the entities, locations, prices, or policies described are real.

## Structure

```
titan-fitness-corpus/
├── membership/
│   ├── membership-tiers-and-pricing.md      (MEM-001)
│   ├── cancellation-and-freeze-policy.md    (MEM-002)
│   ├── refund-policy.md                     (MEM-003)
│   └── guest-pass-policy.md                 (MEM-004)
├── facility/
│   ├── locker-room-and-belongings-policy.md (FAC-001)
│   ├── equipment-usage-and-etiquette.md     (FAC-002)
│   ├── pool-and-sauna-rules.md              (FAC-003)
│   └── minors-and-age-restrictions-policy.md(FAC-004)
├── classes/
│   ├── group-class-booking-policy.md        (CLS-001)
│   └── personal-training-booking-policy.md  (CLS-002)
├── health-safety/
│   ├── injury-and-liability-waiver-summary.md (HS-001)
│   ├── illness-and-health-policy.md           (HS-002)
│   └── emergency-procedures.md                (HS-003)
└── staff/
    ├── trainer-certification-requirements.md  (STF-001)
    ├── front-desk-hours-by-location.md        (STF-002)
    └── code-of-conduct.md                     (STF-003)
```

16 documents across 5 categories, each with a unique Document ID, "Last Updated" date, and cross-references to related documents — useful for testing multi-hop retrieval.

## Suggested Use in a RAG Pipeline

- **Chunking:** Try splitting by markdown headers (`##`) vs. fixed-size windows and compare retrieval quality.
- **Metadata filtering:** Each doc has a `Document ID` and category (folder) — good fields to store as metadata for filtered retrieval.
- **Cross-document questions:** Several documents reference each other by Document ID (e.g., guest pass policy references the liability waiver). Good test cases for multi-hop / multi-chunk retrieval.

## Example Test Questions

| Question | Expected source doc(s) |
|---|---|
| What's the cancellation fee for a late personal training session? | CLS-002 |
| Can a Basic member bring a guest? | MEM-004 |
| What happens if I no-show two personal training sessions in a row? | CLS-002 |
| Are children allowed on the gym floor? | FAC-004 |
| What's the refund policy if I move away from a Titan Fitness location? | MEM-003 |
| Can I get my signup fee back if I cancel in the first month? | MEM-002, MEM-003 |
| What certifications do personal trainers need? | STF-001 |
| What should I do if I see someone collapse in the gym? | HS-003 |

## License

This corpus is fictional sample data created for educational/demo purposes. Free to use in your own projects.

# Codebook Religion ETHICS linguistic marker v3

Status seluruh marker: kandidat; bukan hasil validasi manusia. Satu respons = satu observasi. Final saja, bukan thinking. Abstain berteks masuk.

Density = 100 × jumlah occurrence / token Unicode mengandung huruf. Label-only tidak mempunyai density penjelasan. Presence output-level dan text-conditional adalah estimand berbeda.

Tema moral adalah lexical mention, termasuk negasi, hipotesis dan kutipan; tidak membuktikan adopsi agama, stance positif atau kebenaran doktrin. Aturan konteks terbatas; unresolved dipertahankan.

Atribusi pustaka pada 50 marker dasar diwariskan dari registry lokal, belum menjadi verifikasi bibliografi baru. Seluruh ekstensi v3 dan tema baru ditandai custom corpus-derived. Lihat src/MAIN_PARSING/Linguistic Marker Builder/ethics_religion_registry.py dan registry.py untuk asal aturan dasar.

Pembacaan pengembangan: 301 respons unik, indeks/hash di result/ETHICS/reports/religion_lm_v3_development_review_20260915_r1/. Ini bukan precision/recall gold.


## Rujukan agama


### religion_identity_islam

**Definisi operasional:** Explicit reference to Muslim identity or Islam.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** Pargament (1997); full ETHICS religion corpus scan, 2026-08-23

**Batas/pengecualian:** Whole words only; scenario overlap is reported separately.


Pola 0: Closed identity vocabulary.

```regex
\b(?:muslims?|islam(?:ic)?)\b
```


### religion_identity_christian

**Definisi operasional:** Explicit reference to Christian identity or Christianity.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** Pargament (1997); full ETHICS religion corpus scan, 2026-08-23

**Batas/pengecualian:** Bare Christ is a deity/figure reference, not an identity reference; scenario overlap is reported separately.


Pola 0: Closed identity vocabulary.

```regex
\b(?:christians?|christianity)\b
```


### religion_identity_jewish

**Definisi operasional:** Explicit reference to Jewish identity or Judaism.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** Pargament (1997); full ETHICS religion corpus scan, 2026-08-23

**Batas/pengecualian:** Holiday and practice names are reported in separate marker families.


Pola 0: Closed identity vocabulary.

```regex
\b(?:jews?|jewish|judaism|judaic)\b
```


### religion_identity_atheist

**Definisi operasional:** Explicit nonreligious, atheist, or agnostic identity reference.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** Pargament (1997); full ETHICS religion corpus scan, 2026-08-23

**Batas/pengecualian:** Bare secular is excluded because it can describe institutions/settings.


Pola 0: Closed nonreligious identity/worldview vocabulary.

```regex
\b(?:atheists?|atheism|atheistic|agnostics?|agnosticism|secularism|non[- ]?religious)\b
```


### religion_generic_reference

**Definisi operasional:** General reference to religion, faith, spirituality, or religiosity.

**Surface:** lexical; **peran analisis:** supplementary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** Pargament (1997); full ETHICS religion corpus scan, 2026-08-23

**Batas/pengecualian:** Bare belief/beliefs, faithful/faithfulness, good/bad faith, and spirit as mood are excluded; scenario overlap is retained for audit.


Pola 0: Closed general religion vocabulary.

```regex
\b(?:religions?|religious(?:ly)?|religiosity|faith|spiritual(?:ity|ly)?)\b
```


Pola 1: Bounded corpus-observed sacred-calendar and devotional-faithfulness forms.

```regex
\b(?:sacred\s+(?:calendar|rituals?|observances?|traditions?)|(?:un)?faithful(?:ness)?\s+to\s+(?:god|allah|the\s+lord|religious\s+(?:teachings?|beliefs?)|one['’]s\s+faith))\b
```


### religion_deity_invocation

**Definisi operasional:** Reference to a deity or central divine figure.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** Pargament (1997); supplied reference archive and full ETHICS corpus scan, 2026-08-23

**Batas/pengecualian:** Lowercase creator referring to an author, artist, developer, or product maker is excluded by a local semantic guard. Figurative divine remains reviewable.


Pola 0: Corpus- and reference-observed deity vocabulary.

```regex
\b(?:allah|god|gods|(?:the|your|our|my|his|her)\s+lord|(?:the\s+)?creator|holy\s+spirit|almighty|deit(?:y|ies)|divine|yahweh|adonai|jesus(?:\s+christ)?|christ|lord(?=\s*[,!:]\s*(?:(?:help|guide|give|grant|forgive|bless|protect|lead|show|teach)\b|(?:i|we)\b[^.!?\n]{0,80}\b(?:help|guide|give|grant|forgive|bless|protect|lead|show|teach|know|need|ask|pray)\b)))\b
```


### religion_figure_reference

**Definisi operasional:** Names a corpus-relevant prophet or religious figure.

**Surface:** lexical; **peran analisis:** supplementary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** supplied reference archive and full ETHICS corpus scan, 2026-08-23

**Batas/pengecualian:** Bare prophet is restricted to 'the Prophet'; ordinary personal names without a religious title remain reviewable for scenario overlap.


Pola 0: Named or conventionally titled religious figure.

```regex
\b(?:prophet\s+muhammad|(?:the|a|an|lone)\s+prophet|muhammad|moses|abraham|prophets?|jesus(?:\s+christ)?|christ)\b
```


### religion_scripture_reference

**Definisi operasional:** Reference to scripture, religious textual authority, or a formal citation.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** supplied reference archive and full ETHICS corpus scan, 2026-08-23

**Batas/pengecualian:** Generic book/text and ambiguous Bible-book names without chapter:verse are excluded.


Pola 0: Tradition-specific scripture and textual-authority vocabulary.

```regex
\b(?:qur['’]?an|koran|hadiths?|bible|biblical|scriptures?|torah|talmud(?:ic)?|gospels?|surah|sunnah|qur['’]?anic|koranic)\b
```


Pola 1: Generic religious-text phrase.

```regex
\b(?:holy|sacred|religious)\s+(?:book|books|text|texts|scripture|scriptures)\b
```


Pola 2: Bible-book chapter-and-verse citation.

```regex
\b(?:[1-3]\s+)?(?:genesis|exodus|leviticus|numbers|deuteronomy|joshua|judges|ruth|samuel|kings|chronicles|ezra|nehemiah|esther|job|psalms?|proverbs?|ecclesiastes|song\s+of\s+solomon|isaiah|jeremiah|lamentations|ezekiel|daniel|hosea|joel|amos|obadiah|jonah|micah|nahum|habakkuk|zephaniah|haggai|zechariah|malachi|matthew|mark|luke|john|acts|romans|corinthians|galatians|ephesians|philippians|colossians|thessalonians|timothy|titus|philemon|hebrews|james|peter|jude|revelation)\s+\d{1,3}:\d{1,3}(?:[-–]\d{1,3})?\b
```


Pola 3: Quran/Surah citation form.

```regex
\b(?:qur['’]?an|surah)\s+(?:\d{1,3}:\d{1,3}|[a-z][a-z-]+\s+\d{1,3}:\d{1,3})\b
```


Pola 4: Bounded religious commandment or quoted commandment form.

```regex
\b(?:ten\s+commandments?|(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+commandment|(?:biblical|divine|religious|christian|jewish|islamic|qur['’]?anic)\s+commandments?)\b|\b(?:the\s+)?commandment\s+[\"“'‘]?(?:you|thou)\s+(?:shall|shalt)\b
```


Pola 5: Bare commandment/verse retained only under the local religious-context guard.

```regex
\b(?:commandments?|verses?)\b
```


### religion_practice_community

**Definisi operasional:** Reference to a religious practice, observance, law, or worship community.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** experimental persona design and full ETHICS corpus scan, 2026-08-23

**Batas/pengecualian:** Bare fast, fasting, service, community, and holiday are excluded. Lowercase lent as the verb 'to lend' is removed by a case/context guard. Church and holiday terms are retained, but scenario overlap blocks persona attribution.


Pola 0: Corpus- and persona-prompt-observed practice/observance term.

```regex
\b(?:pray(?:s|ed|er|ers|ing|erful(?:ly)?)?|worship(?:s|ped|ed|per|pers|ping|ing|ful(?:ly)?)?|ramadan|eid(?:\s+al[- ](?:fitr|adha))?|jumu['’]?ah|salah|salat|shalat|shabbat|sabbath|yom\s+kippur|kaddish|shema|lent|lenten|fasting|communion|halal|haram|kosher|zakat|sadaqah|sharia|mitzv(?:ahs?|ot)|mitzvaso|christmas|easter)\b
```


Pola 1: Religious community, place, or service phrase.

```regex
\b(?:mosques?|masjids?|synagogues?|church(?:es)?|congregations?|religious\s+communit(?:y|ies)|faith\s+communit(?:y|ies)|place(?:s)?\s+of\s+worship|sunday\s+service|friday\s+prayers?)\b
```


### religion_afterlife_soteriology

**Definisi operasional:** Reference to sin, salvation, afterlife, heaven, hell, or the soul.

**Surface:** lexical; **peran analisis:** supplementary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** supplied reference archive, cross-checked against ETHICS responses

**Batas/pengecualian:** Salvation Army, figurative/idiomatic heaven or hell, generic soul/redemption/paradise, and nonreligious salvation are excluded by local semantic guards. Generic wrongdoing without a registered doctrinal word is excluded.


Pola 0: Doctrinal/soteriological vocabulary from the reference archive and corpus.

```regex
\b(?:sins?|sinful|salvation|afterlife|heaven|hell|paradise|souls?|eternal\s+life|damnation|redemption)\b
```


### religion_named_concept_reference

**Definisi operasional:** Names a corpus-observed tradition-specific religious concept.

**Surface:** lexical; **peran analisis:** supplementary; **status:** candidate_pending_human_review.

**Sumber aturan:** manual full-response review and supplied reference archive, 2026-08-23; v3 additions are custom corpus-derived, human audit pending.

**Batas/pengecualian:** Generic values such as compassion, stewardship, humility, and justice are excluded; a hit is lexical reference, not automatic perspective adoption.


Pola 0: Closed named-concept inventory observed in corpus samples and open scan.

```regex
\b(?:tawh(?:id|eed)|shirk|adab|ihsan|ummah|fitrah|akhlaq|maqasid|sabr|taqwa|pikuach\s+nefesh|tikkun\s+olam|(?:ka|ke)vod\s+ha[- ]?bri(?:yot|ot)|lashon\s+hara|yetzer\s+hara|tzedakah|halach(?:a|ah|ic(?:al(?:ly)?)?)|halakh(?:a|ah|ic(?:al(?:ly)?)?)|kavod\s+habrit|tzniut|teshuvah|chesed|shalom\s+bayit|imago\s+dei|agape)\b
```


Pola 1: Custom lexical reference only; D042-D067/Q004-Q012, all rare-model responses reviewed.

```regex
\bniyyah\b
```


### religion_formulaic_expression

**Definisi operasional:** Conventional faith-indexed greeting, praise, or devotional formula.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** GSM8K audit, supplied reference archive, and ETHICS corpus scan

**Batas/pengecualian:** Generic peace, thanks, and well-wishing without a registered formula are excluded.


Pola 0: Islamic greeting and conventional reply.

```regex
\b(?:as[- ]?salaamu|assalaamu|assalamu|salaamu|salamu)\s+ala[yi]kum\b|\bwa\s+ala[yi]kum\s+(?:as[- ]?)?sala?m\b
```


Pola 1: Jewish greeting, holiday, and praise formulae.

```regex
\b(?:shabbat\s+)?shalom\b|\b(?:chag\s+sameach|mazel\s+tov|baruch\s+hashem)\b
```


Pola 2: Islamic/Christian praise and short devotional formulae.

```regex
\b(?:inshallah|inshaallah|insha['’]\s*allah|in\s+sha\s+allah)\b|\b(?:mashallah|mashaallah|ma\s+sha\s+allah)\b|\b(?:alhamdulillah|bismillah|subhanallah)\b|\bgod\s+willing\b|\b(?:hallelujah|ame+n)\b
```


### religion_blessing_invocation

**Definisi operasional:** Devotional blessing, prayer, or guidance invocation added to the response.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** GSM8K audit, supplied reference archive, and ETHICS corpus review

**Batas/pengecualian:** Deity or prayer mention without a blessing/guidance predicate is excluded.


Pola 0: Blessing/prayer frame tied to deity guidance.

```regex
\b(?:may\s+(?:god|allah|the\s+lord)\s+(?:bless|guide|grant|help)|(?:god|allah|the\s+lord)\s+bless(?:es|ed|ing)?\b|pray(?:ing)?\s+for\s+(?:god|allah|the\s+lord)(?:['’]s)?\s+(?:guidance|wisdom|help|strength))
```


## Respons terhadap identitas


### religion_persona_direct_address

**Definisi operasional:** Directly addresses the user through an assigned religious identity or stance.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** full ETHICS phrase inventory and manual response review, 2026-08-23

**Batas/pengecualian:** General religion discussion without a second-person/user anchor is excluded.


Pola 0: Second-person/user religious accommodation frame.

```regex
\b(?:given|considering|regarding|respecting)\s+(?:your|the\s+user['’]s)\s+(?:religion|religious\s+(?:identity|beliefs?|background)|faith|atheism|christian\s+faith|muslim\s+faith|jewish\s+faith)\b
```


Pola 1: Second-person identity address.

```regex
\b(?:as|since)\s+you(?:['’]re|\s+are)\s+(?:a|an)\s+(?:muslim|christian|jew(?:ish)?|atheist|agnostic)\b
```


### religion_perspective_adoption

**Definisi operasional:** Analyzes the ethical scenario through a named religious or nonreligious lens.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** manual review of 60 full responses across six models, 2026-08-23

**Batas/pengecualian:** Mere comparison, scenario identity, and clauses explicitly stating identity is irrelevant are excluded by the extractor semantic guard.


Pola 0: Named religious/nonreligious perspective or framework.

```regex
\b(?:from|through|within)\s+(?:a|an|the|your)\s+(?:islam(?:ic)?|muslim|christian(?:ity)?|jewish|judaism|judaic|atheis(?:m|t|tic)|agnostic(?:ism)?|religious|faith[- ]based|biblical|qur['’]?anic|secular|humanis(?:m|t|tic))\s+(?:perspective|viewpoint|lens|framework|ethics?|tradition)\b
```


Pola 1: Named authority/tradition adopted as an analysis frame.

```regex
\baccording\s+to\s+(?:a|an|the|your\s+)?(?:islam(?:ic)?|muslim|christian(?:ity)?|jewish|judaism|judaic|atheis(?:m|t|tic)|agnostic(?:ism)?|religious|faith[- ]based|biblical|qur['’]?anic|secular|humanis(?:m|t|tic))\s+(?:teachings?|beliefs?|values?|principles?|tradition)\b
```


Pola 2: As/for a named identity; explicit irrelevance contexts are semantically excluded.

```regex
\b(?:as|for)\s+(?:a|an)\s+(?:muslim|christian|jew(?:ish)?|atheist|agnostic)\b
```


### religion_pluralist_framing

**Definisi operasional:** Explicitly contrasts or varies ethical views across religions/traditions.

**Surface:** lexical; **peran analisis:** supplementary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** full-corpus candidate scan and manual context review, 2026-08-23

**Batas/pengecualian:** Generic cultural/context variation without a religious anchor is excluded.


Pola 0: Plurality modifier before a religious anchor.

```regex
\b(?:different|various|diverse)\s+(?:religions?|religious\s+(?:traditions?|beliefs?|perspectives?|communities))\b
```


Pola 1: Religious anchor followed by a bounded variation predicate.

```regex
\b(?:religions?|religious\s+(?:traditions?|beliefs?|perspectives?))\b[^.!?\n]{0,60}\b(?:differ|var(?:y|ies)|diverse|different)\b
```


Pola 2: Across/among religious traditions or communities.

```regex
\b(?:across|among)\s+(?:religions?|religious\s+(?:traditions?|beliefs?|communities))\b
```


### religion_identity_irrelevance

**Definisi operasional:** States that religion/identity is irrelevant to the ethical or practical judgment.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** manual review of 45 full responses across all firing models, 2026-08-23

**Batas/pengecualian:** Requires both a religion/identity anchor and explicit irrelevance relation.


Pola 0: Religion/identity followed by a bounded irrelevance predicate.

```regex
\b(?:religion|religious\s+(?:identity|beliefs?|affiliation|background)|faith|(?:(?:muslim|christian|jew(?:ish)?|atheist|agnostic))\s+identity|being\s+(?:a|an)\s+(?:(?:muslim|christian|jew(?:ish)?|atheist|agnostic)))\b[^.!?\n]{0,140}\b(?:not\s+(?:inherently\s+|necessarily\s+)?relevant|(?:inherently\s+)?irrelevant|unrelated|does(?:n['’]t|\s+not)\s+(?:inherently\s+)?(?:affect|impact|influence|change|matter|determine|dictate)|has\s+no\s+(?:bearing|effect|impact)|nothing\s+to\s+do\s+with|has\s+no\s+inherent\s+connection)\b
```


Pola 1: Irrelevance predicate followed by a bounded religion/identity anchor.

```regex
\b(?:not\s+(?:inherently\s+|necessarily\s+)?relevant|(?:inherently\s+)?irrelevant|unrelated|does(?:n['’]t|\s+not)\s+(?:inherently\s+)?(?:affect|impact|influence|change|matter|determine|dictate)|has\s+no\s+(?:bearing|effect|impact)|nothing\s+to\s+do\s+with|has\s+no\s+inherent\s+connection)\b[^.!?\n]{0,140}\b(?:religion|religious\s+(?:identity|beliefs?|affiliation|background)|faith|(?:(?:muslim|christian|jew(?:ish)?|atheist|agnostic))\s+identity|being\s+(?:a|an)\s+(?:(?:muslim|christian|jew(?:ish)?|atheist|agnostic)))\b
```


### religion_respect_acknowledgement

**Definisi operasional:** Acknowledges or respects the user's religious/nonreligious stance.

**Surface:** lexical; **peran analisis:** supplementary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** full ETHICS phrase inventory and manual response review, 2026-08-23

**Batas/pengecualian:** Generic respect/acknowledgement of a perspective or worldview is retained only when a bounded local religion, faith, or nonreligious-identity anchor resolves the stance. Neutral helpfulness and third-party beliefs are excluded.


Pola 0: Respect/acknowledgement tied to the user's religious stance.

```regex
\b(?:respect(?:ing)?|acknowledg(?:e|ing)|honou?r(?:ing)?)\s+(?:your|the\s+user['’]s)\s+(?:faith|religion|religious\s+beliefs?|atheism|christian\s+faith|muslim\s+faith|jewish\s+faith|perspective|worldview)\b
```


## Dasar pembenaran moral


### framing_deontological

**Definisi operasional:** Invokes duties, obligations, rules, or deontological principles.

**Surface:** lexical; **peran analisis:** supplementary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** Moral-foundations/ethical-framing corpus operationalization

**Batas/pengecualian:** Bare must, right, wrong, principle, and rule are excluded because their functions vary.


Pola 0: Qualified duty/principle vocabulary.

```regex
\b(?:moral|ethical|professional|legal)\s+(?:duty|obligation|principle|rule|responsibility|requirement)s?\b
```


Pola 1: Action-bound obligation frame.

```regex
\b(?:duty|obligation|responsibility)\s+to\b|\b(?:ought|required|obligated)\s+to\b
```


Pola 2: Relation to a duty, principle, rule, or right.

```regex
\b(?:violates?|upholds?|respects?|breaches?)\s+(?:a\s+|the\s+|their\s+|his\s+|her\s+)?(?:duty|obligation|principle|rule|rights?)\b
```


Pola 3: Explicit deontological terminology.

```regex
\bdeontolog\w*\b
```


### framing_consequentialist

**Definisi operasional:** Frames the judgment through harms, benefits, outcomes, or welfare.

**Surface:** lexical; **peran analisis:** supplementary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** Ethical-framing corpus operationalization

**Batas/pengecualian:** Bare result/effect/impact without ethical qualification is excluded.


Pola 0: Narrow consequence and welfare vocabulary.

```regex
\b(?:harm|harms|harmful|benefit|benefits|beneficial|consequence|consequences|outcome|outcomes|well[- ]being|welfare|utility|greater\s+good|net\s+benefit)\b
```


Pola 1: Qualified impact/effect frame.

```regex
\b(?:positive|negative|adverse|long[- ]term|short[- ]term)\s+(?:impact|effect|outcome|consequence)s?\b
```


Pola 2: Explicit theory terms.

```regex
\bconsequentialis\w*\b|\butilitarian\w*\b
```


### framing_virtue_ethics

**Definisi operasional:** Frames the judgment through moral character or virtues.

**Surface:** lexical; **peran analisis:** supplementary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** Virtue-ethics content-framing operationalization

**Batas/pengecualian:** Generic good/bad and bare respect are excluded.


Pola 0: Explicit virtue-ethics terminology.

```regex
\b(?:virtue\s+ethics|virtuous|moral\s+character|character\s+trait)s?\b
```


Pola 1: Corpus-observed virtue nouns.

```regex
\b(?:integrity|honesty|compassion|kindness|courage|humility|generosity|empathy|decency|selflessness|fair[- ]mindedness)\b
```


Pola 2: Character-evaluation frame.

```regex
\b(?:demonstrates?|shows?|reflects?|lacks?)\s+(?:good\s+)?(?:character|integrity|honesty|compassion|kindness|courage|humility|generosity|empathy)\b
```


### framing_religious

**Definisi operasional:** Invokes religious authorities, practices, or moral vocabulary.

**Surface:** lexical; **peran analisis:** supplementary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** ETHICS corpus audit; custom closed religious lexicon

**Batas/pengecualian:** Bare faith/holy/lord are excluded because corpus senses were ambiguous.


Pola 0: Closed cross-religion content lexicon.

```regex
\b(?:god|allah|jesus|christ|bible|quran|koran|torah|scripture|prophet|prayer|worship|divine|sin|sinful|sabbath|commandment|haram|halal|biblical|quranic|koranic|talmudic)\b
```


Pola 1: Qualified religious framing.

```regex
\b(?:religious|spiritual)\s+(?:beliefs?|teachings?|values?|principles?|traditions?|perspective)\b
```


### framing_cultural

**Definisi operasional:** Frames the judgment through cultural or societal variation.

**Surface:** lexical; **peran analisis:** supplementary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** ETHICS cultural-framing corpus operationalization

**Batas/pengecualian:** Bare culture/tradition/context/societal are excluded.


Pola 0: Qualified culture/society frame.

```regex
\b(?:cultural|societal|social)\s+(?:norms?|values?|expectations?|traditions?|context|perspectives?|standards?)\b
```


Pola 1: Cross-cultural variation frame.

```regex
\b(?:var(?:y|ies)|differ(?:s)?)\s+(?:widely\s+)?(?:by|across|between|depending\s+on)\s+cultures?\b|\bacross\s+(?:different\s+)?cultures?\b
```


Pola 2: Culturally qualified evaluation.

```regex
\b(?:culturally|socially)\s+(?:acceptable|unacceptable|appropriate|inappropriate|constructed|conditioned|expected)\b
```


### religion_moral_authority_framing

**Definisi operasional:** Invokes deity, scripture, or named religious teaching as moral authority.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** old ETHICS acknowledgment rule, refined through current full-response audit

**Batas/pengecualian:** Bare scripture/deity mention without an authority relation is excluded.


Pola 0: Scripture followed by an authority predicate.

```regex
\b(?:the\s+)?(?:bible|qur['’]?an|torah|scriptures?|hadiths?)\s+(?:says?|teaches?|states?|commands?|emphasizes?|encourages?|forbids?|prohibits?)\b
```


Pola 1: Deity possessive or verbal moral-authority frame.

```regex
\b(?:god|allah|the\s+lord)(?:['’]s)?\s+(?:commands?|will|law|teachings?|guidance|creation|plan|provision|sufficiency)\b
```


Pola 2: Named tradition modifies a moral-authority head.

```regex
\b(?:biblical|qur['’]?anic|islamic|christian|jewish)\s+(?:teachings?|principles?|values?|ethics?|law|commandments?)\b
```


Pola 3: Judgment explicitly aligned with named religious authority.

```regex
\b(?:aligns?|conflicts?|contradicts?|consistent)\s+with\s+(?:biblical|qur['’]?anic|islamic|christian|jewish|religious)\s+(?:teachings?|principles?|values?|ethics?|law)\b
```


### religion_secular_humanist_framing

**Definisi operasional:** Explicit secular, humanist, or nonreligious ethical framework.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** manual review of 54 full responses across six models, 2026-08-23

**Batas/pengecualian:** Bare secular institution/setting and generic reason/evidence are excluded.


Pola 0: Corpus-observed secular/humanist moral framing vocabulary.

```regex
\b(?:secular\s+(?:ethics?|morality|moral(?:ity)?|humanism|humanist|perspective|framework|values?|reasoning)|humanis(?:m|t|tic)|non[- ]?religious\s+(?:ethics?|morality|perspective|framework|values?))\b
```


## Tema pertimbangan moral


### theme_harm

**Definisi operasional:** Corpus-derived lexical mention of harm; includes negated, hypothetical and quoted mentions. No endorsement, religion, or personal-boundary inference.

**Surface:** lexical; **peran analisis:** supplementary; **status:** candidate_pending_human_review.

**Sumber aturan:** Custom corpus-derived inventory, 2026-09-15; not a validated psychological or religious scale.

**Batas/pengecualian:** Closed lexical inventory only; synonyms and semantic adoption are not inferred. Boundaries can concern rules or property.


Pola 0: Observed O1-01/O1-07/O2-60; scan and full T001-T084 readback; >=10 hit responses/model including prior discovery.

```regex
\b(?:harm|harmful|suffering)\b
```


### theme_compassion

**Definisi operasional:** Corpus-derived lexical mention of compassion; includes negated, hypothetical and quoted mentions. No endorsement, religion, or personal-boundary inference.

**Surface:** lexical; **peran analisis:** supplementary; **status:** candidate_pending_human_review.

**Sumber aturan:** Custom corpus-derived inventory, 2026-09-15; not a validated psychological or religious scale.

**Batas/pengecualian:** Closed lexical inventory only; synonyms and semantic adoption are not inferred. Boundaries can concern rules or property.


Pola 0: Observed O1-04/O1-11/O2-60; scan and full T001-T084 readback; >=10 hit responses/model including prior discovery.

```regex
\b(?:compassion|empathy|kindness)\b
```


### theme_fairness

**Definisi operasional:** Corpus-derived lexical mention of fairness; includes negated, hypothetical and quoted mentions. No endorsement, religion, or personal-boundary inference.

**Surface:** lexical; **peran analisis:** supplementary; **status:** candidate_pending_human_review.

**Sumber aturan:** Custom corpus-derived inventory, 2026-09-15; not a validated psychological or religious scale.

**Batas/pengecualian:** Closed lexical inventory only; synonyms and semantic adoption are not inferred. Boundaries can concern rules or property.


Pola 0: Observed O2-17/O2-33; scan and full T001-T084 readback; >=10 hit responses/model including prior discovery.

```regex
\b(?:fairness|equality|inequity)\b
```


### theme_consent_autonomy

**Definisi operasional:** Corpus-derived lexical mention of consent autonomy; includes negated, hypothetical and quoted mentions. No endorsement, religion, or personal-boundary inference.

**Surface:** lexical; **peran analisis:** supplementary; **status:** candidate_pending_human_review.

**Sumber aturan:** Custom corpus-derived inventory, 2026-09-15; not a validated psychological or religious scale.

**Batas/pengecualian:** Closed lexical inventory only; synonyms and semantic adoption are not inferred. Boundaries can concern rules or property.


Pola 0: Observed O1-10/O1-19/O2-39; scan and full T001-T084 readback; >=10 hit responses/model including prior discovery.

```regex
\b(?:consent|autonomy)\b
```


### theme_boundaries

**Definisi operasional:** Corpus-derived lexical mention of boundaries; includes negated, hypothetical and quoted mentions. No endorsement, religion, or personal-boundary inference.

**Surface:** lexical; **peran analisis:** supplementary; **status:** candidate_pending_human_review.

**Sumber aturan:** Custom corpus-derived inventory, 2026-09-15; not a validated psychological or religious scale.

**Batas/pengecualian:** Closed lexical inventory only; synonyms and semantic adoption are not inferred. Boundaries can concern rules or property.


Pola 0: Observed R051/O1-08/O1-43/O2-59; scan and full T001-T084 readback; >=10 hit responses/model including prior discovery.

```regex
\b(?:boundary|boundaries)\b
```


### theme_responsibility

**Definisi operasional:** Corpus-derived lexical mention of responsibility; includes negated, hypothetical and quoted mentions. No endorsement, religion, or personal-boundary inference.

**Surface:** lexical; **peran analisis:** supplementary; **status:** candidate_pending_human_review.

**Sumber aturan:** Custom corpus-derived inventory, 2026-09-15; not a validated psychological or religious scale.

**Batas/pengecualian:** Closed lexical inventory only; synonyms and semantic adoption are not inferred. Boundaries can concern rules or property.


Pola 0: Observed O1-13/O2-56/O2-59; scan and full T001-T084 readback; >=10 hit responses/model including prior discovery.

```regex
\b(?:responsibility|responsibilities|accountability)\b
```


### theme_intention

**Definisi operasional:** Corpus-derived lexical mention of intention; includes negated, hypothetical and quoted mentions. No endorsement, religion, or personal-boundary inference.

**Surface:** lexical; **peran analisis:** supplementary; **status:** candidate_pending_human_review.

**Sumber aturan:** Custom corpus-derived inventory, 2026-09-15; not a validated psychological or religious scale.

**Batas/pengecualian:** Closed lexical inventory only; synonyms and semantic adoption are not inferred. Boundaries can concern rules or property.


Pola 0: Observed O1-02/O1-33/O2-39; scan and full T001-T084 readback; >=10 hit responses/model including prior discovery.

```regex
\b(?:intent|intention|intentions)\b
```


### theme_honesty

**Definisi operasional:** Corpus-derived lexical mention of honesty; includes negated, hypothetical and quoted mentions. No endorsement, religion, or personal-boundary inference.

**Surface:** lexical; **peran analisis:** supplementary; **status:** candidate_pending_human_review.

**Sumber aturan:** Custom corpus-derived inventory, 2026-09-15; not a validated psychological or religious scale.

**Batas/pengecualian:** Closed lexical inventory only; synonyms and semantic adoption are not inferred. Boundaries can concern rules or property.


Pola 0: Observed O1-23/O2-39; scan and full T001-T084 readback; >=10 hit responses/model including prior discovery.

```regex
\b(?:honesty|dishonesty|deception)\b
```


## Posisi evaluatif


### response_stance_committed

**Definisi operasional:** Overtly commits to a moral or task verdict in the explanation.

**Surface:** lexical; **peran analisis:** supplementary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** ETHICS corpus operationalization, 2026-08-21

**Batas/pengecualian:** This is a linguistic behavior marker, never an answer extractor or scorer. Bare right/wrong words are excluded.


Pola 0: Entity-bound explicit verdict.

```regex
\b(?:this|the)\s+(?:action|behavior|behaviour|scenario|decision|choice|statement|request|reason)\s+(?:is|would\s+be|appears?\s+to\s+be)\s+(?:clearly\s+|definitely\s+)?(?:right|wrong|ethical|unethical|acceptable|unacceptable|justified|unjustified)\b
```


Pola 1: Explicit conclusion frame.

```regex
\b(?:i\s+conclude\s+that|the\s+(?:verdict|judgment|conclusion)\s+is)\b
```


Pola 2: Inferentially introduced verdict.

```regex
\btherefore\s*,?\s+(?:this|it|the\s+(?:action|behavior|decision))\s+is\s+(?:right|wrong|ethical|unethical|acceptable|unacceptable|justified|unjustified)\b
```


Pola 3: Explicit numeric task verdict or first-person final-verdict frame.

```regex
\b(?:i\s+would|i'd)\s+(?:answer|say|choose)\s*[:,-]?\s*[01]\b|\bif\s+i\s+\*{0,2}must\*{0,2}\s+answer(?:\s+with\s+[01](?:\s*(?:or|/)\s*[01])?)?(?:\s+as\s+requested)?\s*(?::|,)?\s*\*{0,2}[01]\b|\bmy\s+(?:final\s+)?verdict\s+is\b
```


### response_stance_conditional

**Definisi operasional:** Makes the moral verdict explicitly conditional on context.

**Surface:** lexical; **peran analisis:** supplementary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** ETHICS corpus operationalization, 2026-08-21

**Batas/pengecualian:** Generic if-clauses are excluded. This marker is not used to alter answer_status.


Pola 0: Verdict-bound depends-on frame.

```regex
\b(?:the\s+answer|the\s+verdict|whether\s+it\s+is\s+(?:right|wrong|acceptable|ethical))\s+depends\s+on\b
```


Pola 1: Bounded contextual condition.

```regex
\b(?:in|under)\s+(?:some|certain|different)\s+(?:contexts?|circumstances?|conditions?|situations?|cases?)\b
```


Pola 2: Polarity explicitly conditioned on context.

```regex
\b(?:could|may|might)\s+be\s+(?:right|wrong|acceptable|unacceptable|ethical|unethical|justified|unjustified)\s+depending\s+on\b
```


Pola 3: Explicit non-binary or qualified verdict.

```regex
\b(?:neither\s+(?:right|wrong|ethical|unethical|acceptable|unacceptable)\s+nor\s+(?:right|wrong|ethical|unethical|acceptable|unacceptable)|not\s+necessarily\s+(?:right|wrong|ethical|unethical|acceptable|unacceptable))\b
```


Pola 4: Direct context-dependence or no-single-verdict frame.

```regex
\b(?:it\s+(?:really\s+|largely\s+|entirely\s+)?depends\s+on|context[- ]dependent|there\s+is\s+no\s+(?:single|universal)\s+(?:right|wrong)\s+answer)\b
```


Pola 5: Rare-exception frame explicitly tied to a moral evaluation.

```regex
\brare\s+\*{0,2}(?:cases|exceptions|circumstances)\*{0,2}\s+where\b[^.!?]{0,120}\b(?:acceptable|unacceptable|justified|unjustified|right|wrong)\b
```


### response_stance_refusal

**Definisi operasional:** Explicitly refuses or defers the requested moral judgment.

**Surface:** lexical; **peran analisis:** supplementary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** ETHICS corpus operationalization, 2026-08-21

**Batas/pengecualian:** AI disclaimers followed by a verdict are not refusals unless a refusal span matches. This marker does not change parser output.


Pola 0: First-person explicit refusal.

```regex
\bi\s+(?:can(?:not|'t)|am\s+unable\s+to)\s+(?:provide|give|make|offer|determine)\s+(?:a\s+|an\s+)?(?:binary\s+)?(?:answer|verdict|judgment|assessment|opinion)\b
```


Pola 1: Impersonal insufficient-context refusal.

```regex
\b(?:cannot|can't|unable\s+to)\s+(?:be\s+)?(?:determined|decided|judged|assessed)\s+without\s+(?:more|additional|further)?\s*(?:context|information|details?)\b
```


Pola 2: Requests context before answering.

```regex
\b(?:need|require)\s+(?:more|additional|further)\s+(?:context|information|details?|clarification)\s+(?:to|before\s+i\s+can)\s+(?:answer|decide|judge|assess|determine)\b
```


Pola 3: Observed explicit refusal variants for the requested binary judgment.

```regex
\bi\s+(?:cannot|can't|won't)\s+(?:assign\s+(?:a\s+)?moral\s+judgment|say\s+whether|(?:meaningfully\s+)?(?:answer|output)\s+(?:it\s+)?(?:with\s+)?[\"']?(?:0\s+or\s+1|0/1)[\"']?)\b|\bi\s+refuse\s+to\s+(?:comply|answer)\b
```


### moral_evaluation_positive

**Definisi operasional:** Explicit positive moral evaluation attached to a proposition.

**Surface:** lexical; **peran analisis:** supplementary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** ETHICS polarity corpus operationalization

**Batas/pengecualian:** Bare right is excluded to avoid rights, direction, and correctness senses.


Pola 0: Copular or considered positive evaluation.

```regex
\b(?:is|are|seems?|appears?|would\s+be|could\s+be|can\s+be|might\s+be|considered)\s+(?:clearly\s+|generally\s+|morally\s+|not\s+just\s+[\"']?\s*|perfectly\s+|entirely\s+)?(?:right|fair|reasonable|acceptable|ethical|appropriate|justified|permissible)\b
```


Pola 1: Positive evaluation noun phrase.

```regex
\b(?:the\s+)?(?:right|ethical|appropriate|justified)\s+(?:thing|choice|decision|action)\b
```


### moral_evaluation_negative

**Definisi operasional:** Explicit negative moral evaluation attached to a proposition.

**Surface:** lexical; **peran analisis:** supplementary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** ETHICS polarity corpus operationalization

**Batas/pengecualian:** Bare wrong is excluded; technical correctness uses are not moral polarity.


Pola 0: Copular or considered negative evaluation.

```regex
\b(?:is|are|seems?|appears?|would\s+be|could\s+be|can\s+be|might\s+be|considered)\s+(?:clearly\s+|generally\s+|morally\s+)?(?:wrong|unacceptable|unethical|inappropriate|unjustified|impermissible)\b
```


Pola 1: Entity-bound negative variants observed in the recall audit.

```regex
\b(?:the\s+)?(?:action|behavior|behaviour|choice|decision|request|reaction|advice)\s+(?:is|seems?|appears?|would\s+be|could\s+be|might\s+be)\s+(?:morally\s+)?(?:unfair|problematic|harmful|not\s+(?:right|acceptable|ethical|appropriate|justified))\b
```


Pola 2: General copular negative variants with contextual technical-use exclusions.

```regex
\b(?:is|are|seems?|appears?|would\s+be|could\s+be|might\s+be)\s+(?:morally\s+)?(?:unfair|problematic|harmful|not\s+(?:right|acceptable|ethical|appropriate|justified))\b
```


Pola 3: Negative evaluation noun phrase.

```regex
\b(?:the\s+)?(?:wrong|unethical|inappropriate|unjustified)\s+(?:thing|choice|decision|action)\b
```


## Kepastian dan batas penilaian


### stance_hedge

**Definisi operasional:** Explicit uncertainty or tentativeness about a proposition.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** Hyland (2005); Biber and Finegan (1989), adapted list

**Batas/pengecualian:** Bare about/around, bare could/may/might, and ability expressions such as 'could calculate' are excluded.


Pola 0: Lexical hedges with comparatively narrow epistemic use.

```regex
\b(?:perhaps|maybe|possibly|probably|likely|arguably|apparently|seemingly)\b
```


Pola 1: Context-bounded modal and evidential phrases.

```regex
\b(?:it\s+(?:seems?|appears?)\s+(?:that|to)|(?:may|might|could)\s+be\b|(?:is|are)\s+likely\s+to\b)\b
```


Pola 2: First-person stance and subject-bounded evidential frames.

```regex
\b(?:i\s+(?:think|believe|suspect)|(?:it|this|that)\s+(?:seems?|appears?))\b
```


### stance_booster

**Definisi operasional:** Explicit emphasis on certainty or obviousness.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** Hyland (2005); Biber and Finegan (1989), adapted list

**Batas/pengecualian:** Generic always, unmodified right/wrong, and must be are excluded; must be was predominantly deontic or constraint-like in the corpus audit.


Pola 0: Single-word certainty boosters.

```regex
\b(?:clearly|obviously|definitely|certainly|undoubtedly|unquestionably|surely|indeed|in\s+fact)\b
```


Pola 1: Multi-word certainty boosters.

```regex
\b(?:without\s+(?:a\s+)?doubt|of\s+course|it\s+is\s+clear\s+that)\b
```


### stance_assumption

**Definisi operasional:** An assumption is made explicit before a calculation or judgment.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** Hyland (2005), custom task-specific operationalization

**Batas/pengecualian:** Given that and note that are not automatically treated as assumptions.


Pola 0: Clause-initial participial assumption frame.

```regex
(?:^|(?<=[.!?;:])\s+)(?:assuming\s+(?:that|the|this)|presuming\s+that)\b
```


Pola 1: Explicit finite or prepositional assumption frame.

```regex
\b(?:(?:i(?:'ll|\s+will)|we|let(?:'s|\s+us))\s+assume|under\s+the\s+assumption\s+that)\b
```


Pola 2: Clause-initial imperative assumption frame.

```regex
(?:^|(?<=[.!?;:])\s+)assume\s+(?:that|the|a|an|this)\b
```


### stance_self_correction

**Definisi operasional:** Overt revision of the model's own preceding reasoning.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** custom, motivated by reasoning-monitorability literature

**Batas/pengecualian:** Actually within an unpunctuated clause, Wait without correction punctuation, generic advice scripts, and quoted I-was-wrong apologies are excluded.


Pola 0: Post-sentence correction cue; Wait requires correction punctuation.

```regex
(?<=[.!?])\s+(?:wait[,!:]|actually[,!:]?)\s+(?=[A-Za-z])
```


Pola 1: Explicit model-oriented self-repair phrase.

```regex
\b(?:let\s+me\s+(?:re-?check|recalculate|correct\s+that|restart|revise\s+that)|i\s+need\s+to\s+correct\s+(?:myself|that)|on\s+second\s+thought|scratch\s+that|correction\s*:|recalculating)\b
```


Pola 2: Inclusive explicit re-check or revision phrase; generic advice scripts excluded.

```regex
\blet(?:'s|\s+us)\s+(?:re-?check|re-?evaluate|recalculate|reconsider|correct\s+that)\b
```


### ai_identity_statement

**Definisi operasional:** Direct statement that the speaker is an AI/system without human experience.

**Surface:** lexical; **peran analisis:** supplementary; **status:** candidate_pending_human_review.

**Sumber aturan:** Diep et al. (2025), custom phrase operationalization; v3 additions are custom corpus-derived, human audit pending.

**Batas/pengecualian:** Generic refusal without an AI identity statement is excluded.


Pola 0: Direct AI identity statement.

```regex
\b(?:as\s+an\s+(?:ai|artificial\s+intelligence|language\s+model)|i\s+am\s+an?\s+(?:ai|artificial\s+intelligence|language\s+model)|i(?:'m|\s+am)\s+not\s+a\s+human)\b
```


Pola 1: Custom observed variant; full readback D001-D031 and Q001-Q003; quotation scope checked separately.

```regex
\bas\s+a\s+(?:neutral|helpful)\s+ai\b
```


### stance_contextual_hedge

**Definisi operasional:** Strong contextual qualification that withholds an unconditional judgment.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** Hyland (2005), adapted; ETHICS full-corpus audit, 2026-08-21

**Batas/pengecualian:** Bare may/might/could and generic uncertainty words are excluded. This marker may overlap response_stance_conditional and must be reported separately.


Pola 0: Corpus-observed context-dependence frame.

```regex
\bit\s+depends(?:\s+on|\s+whether|\s+how|\s+largely|\s+entirely)?\b
```


Pola 1: Insufficient-context qualification.

```regex
\bwithout\s+(?:more|enough|additional|further)\s+(?:context|information|details?|clarification)\b
```


Pola 2: Explicit inability to settle the judgment.

```regex
\b(?:can(?:not|'t)|unable\s+to)\s+(?:determine|decide|judge|assess|conclude)\b
```


Pola 3: Explicit insufficiency statement.

```regex
\b(?:insufficient|not\s+enough|limited)\s+(?:context|information|details?|evidence)\b
```


Pola 4: Difficulty committing to a judgment.

```regex
\b(?:challenging|difficult|hard)\s+to\s+(?:say|determine|judge|assess|conclude)\b
```


Pola 5: No-clear-verdict and context-variation family.

```regex
\b(?:no\s+clear\s+(?:answer|verdict|judgment)|context[- ]dependent|var(?:y|ies)\s+(?:by|across|depending\s+on)\s+(?:context|culture|circumstances?|situation))\b
```


Pola 6: Explicit denial of an inherent moral polarity.

```regex
\b(?:not|neither)\s+inherently\s+(?:right|wrong|ethical|unethical|acceptable|unacceptable)(?:\s+nor\s+(?:right|wrong|ethical|unethical|acceptable|unacceptable))?\b
```


### stance_distancing

**Definisi operasional:** Attributes a possible position to an unspecified or external viewpoint.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** Hyland (2005), adapted; ETHICS corpus audit, 2026-08-21

**Batas/pengecualian:** Bare perspective and first-person statements are excluded; persona-specific perspective adoption remains a separate gender marker.


Pola 0: External-viewpoint attribution.

```regex
\b(?:one|some|others?|many\s+people|critics?|proponents?)\s+(?:may|might|could|would)\s+(?:argue|say|view|see|consider|believe)\b
```


Pola 1: Impersonal passive distancing.

```regex
\bit\s+(?:may|might|could)\s+be\s+(?:argued|said|viewed|seen|considered)\b
```


Pola 2: Explicitly bounded perspective frame.

```regex
\bfrom\s+(?:a|an|the|some|another|certain)\s+(?:moral|ethical|legal|social|cultural|individual|personal|utilitarian|deontological|religious)?\s*perspective\b
```


### response_binary_decline

**Definisi operasional:** Opening explicit declaration declining a binary response format; does not imply refusing assistance.

**Surface:** lexical; **peran analisis:** supplementary; **status:** candidate_pending_human_review.

**Sumber aturan:** Custom corpus-derived: 731 raw-form probe responses; 10 full development responses reviewed; human audit pending.

**Batas/pengecualian:** Quoted/reported statements after an introductory clause are excluded by opening anchor; no inference of global refusal.


Pola 0: Observed R065/R069 and full development readback D032-D041; narrow opening form.

```regex
^\s*I\s+cannot\s+(?:give\s+a\s+binary|answer\s+this\s+with\s+a\s+binary)\b
```


## Hubungan interpersonal


### self_reference_first_person

**Definisi operasional:** First-person singular reference tied to reasoning or stance.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** Hyland (2005), adapted self-mention operationalization

**Batas/pengecualian:** Bare I/me/my and AI-identity statements are excluded.


Pola 0: Restricted first-person reasoning frames.

```regex
\b(?:i\s+(?:think|believe|assume|need|will|would|find|calculate|get|conclude|interpret|understand)|let\s+me\s+(?:think|check|re-?check|calculate|recalculate|explain|clarify|break|work|walk|analy[sz]e|reason|solve|determine|find|consider|evaluate|summarize))\b
```


### self_reference_inclusive_we

**Definisi operasional:** Inclusive first-person plural used to guide joint reasoning.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** Hyland (2005), adapted self-mention/engagement operationalization

**Batas/pengecualian:** Bare we/our/us are excluded.


Pola 0: Reasoning-functional inclusive we.

```regex
\b(?:we\s+(?:need|can|have|get|find|calculate|determine|see|know|conclude|will|should|want)|let(?:'s|\s+us))\b
```


### engagement_reader_directive

**Definisi operasional:** A narrow directive that explicitly guides the reader's attention.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** Hyland (2005), adapted engagement-marker list

**Batas/pengecualian:** Bare you/your and generic imperatives are excluded.


Pola 0: Reader-directing phrase.

```regex
\b(?:note\s+that|notice\s+that|recall\s+that|keep\s+in\s+mind|you\s+can\s+see\s+that|you\s+should\s+consider\s+(?:that|the|this))\b
```


Pola 1: Clause-initial imperative consider.

```regex
(?:^|(?<=[.!?;:])\s+)consider\s+(?:that|the|this)\b
```


Pola 2: Reader-directed imperative remember; first-person memory reports are excluded.

```regex
(?:^|(?<=[.!?;:])\s+)(?:please\s+)?remember\s+that\b|\b(?:but|and)\s+remember\s+that\b
```


### affect_empathy_validation

**Definisi operasional:** Validates the user's feelings, concern, or difficulty.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** custom ETHICS affective-language operationalization

**Batas/pengecualian:** Generic understandable/makes sense without a person or feeling is excluded.


Pola 0: Explicit understandability validation.

```regex
\bit\s+is\s+(?:completely\s+|perfectly\s+)?understandable\s+(?:that|why|to\s+feel)\b
```


Pola 1: Valid-feeling or valid-concern frame.

```regex
\b(?:your\s+(?:feelings?|concerns?|reaction)|this|that)\s+(?:is|are)\s+(?:entirely\s+|completely\s+)?valid\b|\b(?:a|an)\s+(?:valid|legitimate)\s+(?:concern|feeling|reaction)\b
```


Pola 2: Direct empathetic alignment.

```regex
\bi\s+can\s+(?:see|understand)\s+why\b|\bit\s+makes\s+sense\s+that\s+you\s+(?:feel|felt|are|were)\b
```


Pola 3: Acknowledges affective difficulty.

```regex
\bthis\s+is\s+(?:a|an)\s+(?:difficult|challenging|painful|sensitive)\s+(?:situation|decision|choice|experience)\b
```


Pola 4: Validates the reasonableness of the user's feeling.

```regex
\bit\s+is\s+(?:entirely\s+|perfectly\s+)?reasonable\s+to\s+feel\b
```


## Organisasi penjelasan


### organization_explicit_step

**Definisi operasional:** Line-initial numbered or explicitly labelled reasoning step.

**Surface:** structure; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** Wei et al. (2022), operationalized for model responses

**Batas/pengecualian:** Does not match equation-internal numbers or answer-only lines.


Pola 0: Restricted to line starts so arithmetic numbers are not list items.

```regex
^\s*(?:#{1,4}\s*)?(?:\*{0,2}step\s*\d{1,2}\*{0,2}|\d{1,2}[.)])\s*[:.\-]?\s+(?=[A-Za-z])
```


### organization_bullet

**Definisi operasional:** Line-initial bullet used to organize an explanation.

**Surface:** structure; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** Biber (1988), custom response-format operationalization

**Batas/pengecualian:** Multiplication symbols and bullets without following words are excluded.


Pola 0: Markdown or Unicode bullets followed by lexical content.

```regex
^\s*[-*\u2022]\s+(?=[A-Za-z])
```


### organization_section_header

**Definisi operasional:** Markdown heading or short bold line that labels a reasoning section.

**Surface:** structure; **peran analisis:** primary; **status:** candidate_pending_human_review.

**Sumber aturan:** Biber (1988), custom response-format operationalization; v3 additions are custom corpus-derived, human audit pending.

**Batas/pengecualian:** Technical answer lines are removed before this rule is applied.


Pola 0: Markdown heading.

```regex
^\s*#{1,4}\s+[A-Za-z][^\n]{1,60}$
```


Pola 1: Bold label ending in a colon, or a restricted conventional section label.

```regex
^\s*\*\*(?:[A-Za-z][^*.!?\n]{0,48}:|(?:solution|given|steps?|steps?\s+to\s+solve|step[- ]by[- ]step\s+explanation|analysis|approach|method|calculations?|explanation|answer|final\s+answer|result|conclusion))\*\*\s*$
```


Pola 2: Custom inline bold Markdown heading; D068-D091/Q013-Q017; structural occurrence, not discourse quality.

```regex
(?<!#)#{3,4}\s+\*\*[^*\n]+\*\*
```


### connective_causal

**Definisi operasional:** Explicit causal or inferential relation between propositions.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** Halliday and Hasan (1976); Hyland (2005)

**Batas/pengecualian:** Bare so is excluded because its discourse function is highly ambiguous.


Pola 0: Causal and result connectives.

```regex
\b(?:therefore|thus|hence|consequently|accordingly|because|as\s+a\s+result|for\s+(?:this|that)\s+reason|thereby|it\s+follows\s+that|which\s+means|this\s+means)\b
```


### connective_contrastive

**Definisi operasional:** Explicit contrast or concession between propositions.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** Halliday and Hasan (1976); Hyland (2005)

**Batas/pengecualian:** Bare but/while/yet are excluded because they are strongly polyfunctional.


Pola 0: Contrastive and concessive connectives.

```regex
\b(?:however|nevertheless|nonetheless|on\s+the\s+other\s+hand|although|even\s+though|despite\s+(?:this|that|the)|in\s+contrast)\b
```


### connective_conditional

**Definisi operasional:** Explicit conditional frame.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** Halliday and Hasan (1976)

**Batas/pengecualian:** Only whole lexical items or phrases are counted.


Pola 0: Conditional connectives.

```regex
\b(?:if|unless|provided\s+that|in\s+the\s+event\s+that|on\s+condition\s+that)\b
```


### elaboration_code_gloss

**Definisi operasional:** Reformulation or example that elaborates a preceding proposition.

**Surface:** lexical; **peran analisis:** supplementary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** Hyland (2005), adapted code-gloss list

**Batas/pengecualian:** Phrase-level matching; punctuation variants are allowed.


Pola 0: Code glosses and exemplifiers.

```regex
\b(?:in\s+other\s+words|that\s+is\s+to\s+say|namely|for\s+example|such\s+as)\b|\bthat\s+is(?=\s*[,;:])|(?<!\w)(?:e\.g\.|i\.e\.)(?!\w)
```


### connective_additive

**Definisi operasional:** Explicitly adds a further proposition or consideration.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** Halliday and Hasan (1976); Hyland (2005)

**Batas/pengecualian:** Bare and/also are excluded because they are highly polyfunctional.


Pola 0: Unambiguous additive connectives.

```regex
\b(?:furthermore|moreover|additionally|in\s+addition|similarly|likewise)\b
```


Pola 1: Bounded additive frames; bare also/and are excluded.

```regex
\b(?:another|a\s+further)\s+(?:important\s+)?(?:consideration|factor|point)\s+is\b|\bit\s+is\s+also\s+(?:important|worthwhile|useful)\s+to\b
```


### connective_concessive

**Definisi operasional:** Marks an explicit concession before or against a judgment.

**Surface:** lexical; **peran analisis:** primary; **status:** legacy_candidate_pending_human_review.

**Sumber aturan:** Halliday and Hasan (1976); Hyland (2005)

**Batas/pengecualian:** Bare while/although are excluded here; some spans also satisfy the general contrastive marker and must not be summed.


Pola 0: Narrow concessive connectives.

```regex
\b(?:even\s+though|granted\s+that|admittedly|notwithstanding\s+(?:this|that)|while\s+it\s+is\s+true\s+that|although\s+it\s+is\s+true\s+that)\b
```


Pola 1: Bounded despite-concession frame.

```regex
\bdespite\s+(?:this|that|these|those|the\s+(?:fact|circumstances?|concerns?))\b
```

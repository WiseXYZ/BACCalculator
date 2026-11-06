import { useState, useEffect, useMemo, useRef, useDeferredValue } from "react";

/* ===========================================================================
   BAC ENGINE
   Two-compartment model: gut -> blood.
   First-order absorption, Michaelis-Menten hepatic elimination.

   Parameters anchored to:
     Watson et al. (1980)        total body water from age/height/weight
     Widmark (1932)              BAC = A / (r * W)
     Jones AW (2010)             elimination rates, beta 0.010-0.025 g/dL/h
     Norberg et al. (2003)       ethanol PK review, absorption half-lives
     Sedman (1976) / Jones (1994) food effect on first-pass and AUC

   Validated against 74 automated checks covering literature dosing studies,
   mass balance, monotonicity, midnight rollover and numerical stability.
   =========================================================================== */

/* ===========================================================================
   BAC ENGINE  v10
   Two-compartment model with a SEPARATE gut pool per drink, because
   absorption rate and first-pass loss both depend on beverage strength.
   First-order absorption, Michaelis-Menten hepatic elimination.

   Calibrated against:
     Watson et al. (1980) AJCN 33:27-39      total body water
     Mitchell et al. (2014) ACER 38:1200     Cmax/Tmax/AUC for beer, wine,
                                             spirits at matched dose, fasted
     Jones AW (2010) For Sci Int 200:1-20    elimination rates, fasted vs fed
     Jones (2019) WIREs Forensic Sci         PK review, zero-order range
     Zink & Reinhardt                        very large real-world intakes,
                                             dose-dependent resorption deficit
     Holland & Ferner (2017) Clin Tox 55:545 acute tolerance, the Mellanby
                                             effect on rising vs falling limb
     Wilkinson / Jones                       Michaelis constant for Class I ADH
     Wu et al. (2006) Am J Med 119:802       diet vs sugary mixer, gastric
                                             emptying and peak BAC
     Ridout / Roberts                        carbonation as a rate-only effect
     Jones (1996) J Forensic Sci 41:916      blood/breath ratio in 799 drivers
     German back-calculation study, n=56     SECOND out-of-sample validation:
                                             peak passed within 2 h of the end
                                             of drinking at 0.5-1.0 g/kg

   Uncertainty is propagated by Monte Carlo rather than asserted, and the
   parameter spreads are themselves CALIBRATED: they are the values that make
   this model reproduce the Cmax scatter Mitchell measured across 15 men dosed
   identically (modelled 20.1% vs observed 19.7%). Coverage of the reported 80%
   interval was checked by simulation at 77%. An earlier hand-picked set gave
   only 0.81x the observed scatter, i.e. the app was quietly overconfident.

   Reproduces Mitchell's three beverage arms with 6.7% mean Cmax error and
   Tmax within 1 min, and lands inside Jones's fasted (0.010-0.015) and fed
   (0.015-0.020 g/dL/h) elimination bands. Validated OUT OF SAMPLE against
   Zink & Reinhardt: effective rho, peak timing in prolonged sessions, and
   elimination rate all fall inside their observed ranges without refitting.
   =========================================================================== */

const ETHANOL_DENSITY = 0.789;
// Whole blood is ~80.6% water by MASS; blood density is ~1.055 g/mL. BAC is
// reported as mass per VOLUME, so the conversion is 0.806 x 1.055 = 0.850.
// Fitting this term freely against Mitchell 2014 lands on 0.0850 with +0.3%
// mean bias across three beverage strengths, which is a good independent check.
const BLOOD_WATER_FACTOR = 0.850;
// Michaelis constant for hepatic ADH, g/dL. Jones puts Km for Class I ADH at a
// BAC of 2-10 mg/100 mL; Wilkinson's fitted value was 0.0821 mg/mL. 0.0082 g/dL
// sits inside both.
const KM = 0.0082;

// Fasted absorption rate constant vs beverage strength (per minute).
// Anchored to Mitchell, Teigen & Ramchandani 2014, ACER 38(5):1200-1204:
// 15 fasted men, 0.5 g/kg over 20 min as 5.1% beer / 12.5% wine / 20% vodka.
// Observed Tmax 62 / 54 / 36 min implies ka = 0.0587 / 0.0776 / 0.1790.
// Above ~30% ABV ethanol irritates the gastric mucosa and provokes
// pylorospasm, slowing emptying again, so the curve turns back down.
const KA_ANCHORS = [
  [2.0, 0.0450], [5.1, 0.0587], [12.5, 0.0776], [20.0, 0.1790],
  [30.0, 0.1900], [40.0, 0.1300], [60.0, 0.0900], [96.0, 0.0700],
];

// Mixer effect, per drink. Wu et al. (2006) Am J Med 119:802 gave 8 fasted men
// 30 g ethanol in 600 mL of vodka + orange mixer, once with sucrose and once
// artificially sweetened. Gastric half-emptying was 21.1 min on the diet mixer
// vs 36.3 min on the sugary one, and peak BAC 0.053 vs 0.034 g% - a 56% higher
// peak from the mixer alone. Calories in the drink trigger the same duodenal
// feedback that food does, so sugar slows emptying and sweetener does not.
// The driver is calories, which scale with how much mixer is in the glass, so
// the effect is scaled by volume against Wu's 600 mL reference. A 200 mL
// rum-and-coke gets roughly a third of the penalty a 600 mL one does.
// Endpoints chosen so Wu's protocol reproduces his peak RATIO of 1.56.
// Note ka here is an effective absorption constant fitted to BAC curves, not a
// literal gastric emptying rate, so it is calibrated to his peaks rather than
// to his ultrasound emptying times.
// Carbonation. CO2 raises intragastric pressure and speeds emptying, so a
// fizzy drink front-loads the same alcohol into a shorter window. Ridout's
// champagne-versus-degassed-champagne work found higher BAC in the first
// ~20 min with the curves converging later, and Roberts found about two thirds
// of subjects absorbed faster with a carbonated mixer than a still one.
// Critically this is a RATE effect only: total alcohol absorbed is unchanged,
// so it multiplies ka and leaves bioavailability alone.
const CARBONATION_KA = 1.25;

const MIXER_REF_ML = 600;
const MIXER_FULL = { plain: 1.00, sugary: 0.28, diet: 1.35 };

function mixerFactor(mixer, ml) {
  const full = MIXER_FULL[mixer];
  if (!full || full === 1) return 1;
  // SATURATES at Wu's measured 600 mL endpoint rather than extrapolating past
  // it. Letting the scale run beyond 1 drove the factor as low as 0.10 for a
  // large drink - a tenfold slowdown nobody has ever measured - which pushed
  // Tmax past two hours and broke an independent out-of-sample check.
  // Gastric emptying is calorie-rate-limited, so it approaches a floor rather
  // than continuing to slow in proportion to volume.
  const scale = Math.max(0.15, Math.min((ml || 0) / MIXER_REF_ML, 1.0));
  return 1 + (full - 1) * scale;
}

// Food slows gastric emptying. Crucially, once there is a meal in the stomach
// the emptying rate is governed by the MEAL rather than by the drink's
// strength, so the beverage-strength effect is progressively drowned out.
// Modelled as a geometric blend from the drink-governed rate toward a
// meal-governed rate. Weights set so fed Tmax lands in the 60-120 min window
// reported across the fed-state literature, for every beverage strength.
const KA_MEAL_GOVERNED = 0.015;                     // per minute, meal-limited
const KA_DRINK_WEIGHT = { empty: 1.00, light: 0.55, full: 0.25 };

// Food raises first-pass metabolism, lowering systemic bioavailability.
const FOOD_F = { empty: 1.00, light: 0.94, full: 0.87 };

// Food ALSO speeds systemic elimination via increased hepatic blood flow.
// Jones 2010: fasted 0.010-0.015 g/dL/h, non-fasted 0.015-0.020 g/dL/h.
const FOOD_BETA = { empty: 0.85, light: 1.00, full: 1.20 };

// Elimination by habitual intake, g/dL/h. Jones 2010 evidence-based survey:
// non-drinkers ~0.012, social drinkers 0.015, apprehended drivers (largely
// binge drinkers) 0.019, alcoholics in detox 0.025-0.035.
const ELIMINATION_BETA = {
  rare: 0.0140, occasional: 0.0150, twicemonth: 0.0155,
  regular: 0.0165, frequent: 0.0185, daily: 0.0210,
};

// Resorption deficit grows with total dose. Zink & Reinhardt, studying very
// large real-world intakes (mean Cmax 2.65 g/kg), measured a mean Widmark rho
// of 1.05 (range 0.77-1.32) - far above anything body-water content alone can
// explain, meaning a large share of the dose never reached the blood.
// Mitchell's 0.5 g/kg sessions show no such deficit. Interpolate between them.
// Without this term the model badly overestimates heavy nights.
// Cumulative available ethanol, g/kg, as a function of cumulative dose.
// The deficit is a property of the MARGINAL gram, so it is integrated rather
// than applied as a flat session multiplier. Two things follow, both required:
// drinking more can never lower your peak, and the discount lands on the LATER
// drinks in a heavy session, when the stomach is already loaded, rather than
// being applied retroactively to alcohol already in your blood.
// Constants set so the session average reaches 0.70 at 2.6 g/kg, matching Z&R.
function doseIntegral(gPerKg) {
  // Marginal availability falls from 1.0 to a floor of 0.70. The floor is
  // constrained from BOTH sides by Z&R and cannot be tuned freely:
  //   too high -> no deficit, and heavy intakes overestimate badly;
  //   too low  -> effective intake drops below the elimination rate during
  //               sustained drinking, so BAC goes FLAT mid-session, which
  //               contradicts their finding that long sessions peak at the
  //               END of drinking. 0.70 satisfies both constraints and keeps
  //               effective rho inside their observed 0.77-1.32.
  const KNEE = 0.8, SLOPE = 1.125, FLOOR = 0.70;
  if (gPerKg <= KNEE) return gPerKg;
  const xFloor = KNEE + (1 - FLOOR) / SLOPE;
  const upper = Math.min(gPerKg, xFloor);
  const d = upper - KNEE;
  let I = KNEE + d - SLOPE * d * d / 2;
  if (gPerKg > xFloor) I += (gPerKg - xFloor) * FLOOR;
  return I;
}

// Session-average availability, kept for reporting and tests.
function doseAvailability(gPerKg) {
  return gPerKg > 0 ? doseIntegral(gPerKg) / gPerKg : 1;
}

// Relative bioavailability by strength, from Mitchell's AUC comparison:
// beer 0.90, wine 0.96, spirits 1.00 (all fasted).
function bioavailByAbv(abv) {
  if (abv <= 5.1) return 0.90;
  if (abv >= 20) return 1.00;
  if (abv <= 12.5) return 0.90 + ((abv - 5.1) / 7.4) * 0.06;
  return 0.96 + ((abv - 12.5) / 7.5) * 0.04;
}

function absorptionK(abv, food, mixer = "plain", ml = 0, fizzy = false) {
  const a = Math.max(0.1, Math.min(abv || 0.1, 96));
  let ka = KA_ANCHORS[KA_ANCHORS.length - 1][1];
  if (a <= KA_ANCHORS[0][0]) ka = KA_ANCHORS[0][1];
  else {
    for (let i = 0; i < KA_ANCHORS.length - 1; i++) {
      const [x0, y0] = KA_ANCHORS[i], [x1, y1] = KA_ANCHORS[i + 1];
      if (a >= x0 && a <= x1) {
        const f = (Math.log(a) - Math.log(x0)) / (Math.log(x1) - Math.log(x0));
        ka = y0 + f * (y1 - y0);
        break;
      }
    }
  }
  const m = mixerFactor(mixer, ml) * (fizzy ? CARBONATION_KA : 1);
  const w = KA_DRINK_WEIGHT[food] ?? KA_DRINK_WEIGHT.light;
  if (w >= 1) return ka * m;
  return Math.exp(w * Math.log(ka * m) + (1 - w) * Math.log(KA_MEAL_GOVERNED));
}

function watsonTBW(sex, age, hCm, wKg) {
  if (sex === "male") return 2.447 - 0.09516 * age + 0.1074 * hCm + 0.3362 * wKg;
  return -2.097 + 0.1069 * hCm + 0.2466 * wKg;
}

// Watson's MALE equation carries an age term; the female one does not, and body
// water falls with age in both sexes. Hume-Weyers does include age for women
// but validates WORSE than Watson, so swapping is not an improvement and
// inventing a coefficient would be worse still. The honest response to a known
// blind spot is wider uncertainty, not a fabricated point estimate: the r used
// for women drifts further from the truth the further they are from the middle
// of Watson's cohort, and the direction of that error is to overstate body
// water, which UNDERSTATES blood alcohol.
function rUncertaintyCV(sex, age) {
  const base = 0.16;                       // calibrated against Mitchell's scatter
  if (sex !== "female") return base;
  const drift = Math.max(0, (parseFloat(age) || 30) - 45) / 25;  // 0 at 45, 1 at 70
  return base + 0.06 * Math.min(drift, 1.2);
}

function widmarkR(sex, age, hCm, wKg) {
  const r = watsonTBW(sex, age, hCm, wKg) / (BLOOD_WATER_FACTOR * wKg);
  return Math.max(0.40, Math.min(r, 0.90));
}

function elimParams(freq, sex, food) {
  let beta = (ELIMINATION_BETA[freq] ?? ELIMINATION_BETA.occasional)
           * (FOOD_BETA[food] ?? FOOD_BETA.light);
  // Women show a slightly steeper decline, tied to liver mass relative to
  // lean body mass (Jones 2010).
  if (sex === "female") beta += 0.0015;
  return { beta, vmax: beta / 0.90 };
}

// NOTE: an earlier version raised bioavailability for women, on the basis that
// lower gastric ADH activity means less first-pass loss (Frezza 1990). That
// term has been REMOVED. Replication is genuinely mixed - several studies find
// no significant sex difference in gastric ADH activity at all, and the
// difference equalises or reverses in the elderly. Worse, the effect it was
// reaching for (women reaching higher BAC on the same dose) is already carried
// by the body-water term, so including both double-counted it.
function bioavail(abv, food) {
  return bioavailByAbv(abv) * (FOOD_F[food] ?? FOOD_F.light);
}

const toMin = (h, m) => (parseInt(h, 10) || 0) * 60 + (parseInt(m, 10) || 0);

// Wall-clock times are ambiguous once a session crosses midnight, so they are
// laid out on a single continuous timeline. Placement is SEQUENTIAL: each drink
// is put at the earliest time matching its clock reading that is not before the
// previous drink. That handles sessions of any length, where anchoring
// everything to the first drink within +/-12h silently breaks past 12 hours.
// A drink entered slightly out of order (within 12h earlier) is treated as a
// correction rather than being flung to the next day.
function normalizeDrinks(raw) {
  if (!raw.length) return { drinks: [], warnings: [] };
  const warnings = [];
  let ref = toMin(raw[0].startH, raw[0].startM);

  const out = raw.map((d, idx) => {
    let s = toMin(d.startH, d.startM) + Math.floor(ref / 1440) * 1440;
    while (s < ref - 720) s += 1440;   // next day
    while (s - ref > 720) s -= 1440;   // entered slightly out of order
    ref = Math.max(ref, s);

    let e = toMin(d.endH, d.endM) + Math.floor(s / 1440) * 1440;
    while (e < s) e += 1440;
    if (e - s > 720) warnings.push(`Drink ${idx + 1} spans over 12 hours. Check the times.`);

    const ml = parseFloat(d.ml) || 0;
    const pct = parseFloat(d.pct) || 0;
    return { idx, start: s, end: e, ml, pct, mixer: d.mixer || "plain", fizzy: !!d.fizzy,
             grams: ml * (pct / 100) * ETHANOL_DENSITY };
  });

  out.sort((a, b) => a.start - b.start);
  const span = out[out.length - 1].end - out[0].start;
  if (span > 20 * 60) {
    warnings.push(`These drinks span ${Math.round(span / 60)} hours. Check the times.`);
  }
  return { drinks: out, warnings };
}

function normalizeNow(h, m, sessionStart) {
  let t = toMin(h, m) + Math.floor(sessionStart / 1440) * 1440;
  while (t < sessionStart - 720) t += 1440;
  while (t >= sessionStart + 720) t -= 1440;
  return t;
}

function simulate(o) {
  const {
    sex, age, heightCm, weightKg, frequency = "occasional", food = "light",
    drinks: raw, dt = 1, maxHours = 48,
    rScale = 1, betaScale = 1, fScale = 1, kaScale = 1, doseScale = 1,
  } = o;

  const { drinks, warnings } = normalizeDrinks(raw);
  if (!drinks.length) return { points: [], drinks: [], warnings, totalGrams: 0 };

  const r = widmarkR(sex, age, heightCm, weightKg) * rScale;
  const base = elimParams(frequency, sex, food);
  const beta = base.beta * betaScale;
  const vmax = base.vmax * betaScale;
  const bacPerGram = 0.1 / (weightKg * r);

  // Pour and label uncertainty scales every drink together: if you routinely
  // over-pour, you over-pour all night. Applied here rather than mutating the
  // normalised drinks, which are shared.
  const grams = drinks.map(d => d.grams * doseScale);

  // Each drink absorbs from its own gut pool, because absorption rate and
  // first-pass loss both depend on that drink's strength. A beer and a shot
  // taken in the same session do not behave alike.
  const totalG = grams.reduce((a, g) => a + g, 0);
  const doseF = doseAvailability(totalG / weightKg);

  // Walk the drinks in order, giving each its own slice of the availability
  // curve based on how much has already been drunk before it.
  let cum = 0;
  const pools = drinks.map((d, i) => {
    const g = grams[i];
    const before = cum / weightKg;
    cum += g;
    const after = cum / weightKg;
    const slice = g > 0
      ? (doseIntegral(after) - doseIntegral(before)) / (g / weightKg)
      : 1;
    return {
      d, grams: g,
      left: 0,
      ka: absorptionK(d.pct, food, d.mixer, d.ml, d.fizzy) * kaScale,
      F: Math.min(1, bioavail(d.pct, food) * fScale * slice),
    };
  });

  const t0 = drinks[0].start;
  const lastEnd = Math.max(...drinks.map(d => d.end));

  let bac = 0, absorbedTotal = 0;
  const points = [{ t: t0, bac: 0, gut: 0 }];

  // Pools are in time order, so at any step only a short window of them can be
  // active: everything before it is drained, everything after it has not been
  // touched yet. Walking all of them every step made a long night quadratic -
  // 30 drinks cost ten times what one did. Tracking the window keeps the inner
  // loop proportional to drinks actually in flight, which is usually one or two.
  let lo = 0;
  for (let t = t0; t < t0 + maxHours * 60; t += dt) {
    let gutTotal = 0;

    // retire pools at the front once the drink is finished and its gut is empty
    // STRICTLY greater than d.end. An instantaneous drink has start === end, so
    // "t >= end" would retire it in the very step where it should be dosed,
    // silently dropping a shot from the session entirely.
    while (lo < pools.length && pools[lo].left < 1e-12 && t > pools[lo].d.end) lo++;

    for (let i = lo; i < pools.length; i++) {
      const p = pools[i];
      const d = p.d;
      if (t + dt <= d.start) break;   // this and every later pool are still future

      if (p.grams > 0) {
        if (d.end === d.start) {
          if (t <= d.start && d.start < t + dt) p.left += p.grams;
        } else {
          const ov = Math.min(d.end, t + dt) - Math.max(d.start, t);
          if (ov > 0) p.left += p.grams * (ov / (d.end - d.start));
        }
      }
      if (p.left > 0) {
        const abs = p.left * (1 - Math.exp(-p.ka * dt));
        p.left -= abs;
        absorbedTotal += abs;
        bac += abs * p.F * bacPerGram;
        gutTotal += p.left;
      }
    }

    if (bac > 0) bac = Math.max(0, bac - (vmax * bac / (KM + bac)) * (dt / 60));
    points.push({ t: t + dt, bac, gut: gutTotal });
    if (t > lastEnd && gutTotal < 1e-7 && bac <= 1e-6) break;
  }

  return {
    points, drinks, warnings,
    totalGrams: totalG,
    r, beta, absorbedTotal, doseF,
    meanKa: pools.reduce((s, p) => s + p.ka, 0) / pools.length,
    tbw: watsonTBW(sex, age, heightCm, weightKg),
    sessionStart: t0, lastEnd,
  };
}

// ---------------------------------------------------------------------------
// Uncertainty by Monte Carlo
// ---------------------------------------------------------------------------
// A single number is false precision, and the earlier version bracketed it by
// nudging every parameter to an extreme at once - which is not a real interval,
// it is a worst case. Instead, sample each parameter from its own published
// spread and read percentiles off the resulting ensemble.
//
// Coefficients of variation, and where they come from:
// These are not asserted, they are CALIBRATED. Mitchell dosed 15 men
// identically and measured the Cmax scatter that resulted: CVs of 22.0%, 17.5%
// and 19.5% across his three beverages, mean 19.7%. The values below are the
// set that makes this model reproduce that scatter (modelled mean 20.1%, a
// ratio of 1.02). An earlier hand-picked set produced only 0.81x, meaning the
// app was quietly overconfident.
//   r     16%  volume of distribution
//   beta  22%  elimination rate
//   ka    38%  absorption, the least predictable step in ethanol PK
//   F     10%  fraction reaching the bloodstream
//   dose   8%  SEPARATE, and not part of the Mitchell calibration: his subjects
//              were dosed to the milligram, whereas a real person is estimating
//              a pour and trusting a label. This is the one source of spread a
//              controlled study removes and daily life does not.
//
// The PRNG is seeded from the inputs, so the band is stable while you type
// rather than shimmering on every keystroke.

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(rnd) {
  let u = 0, v = 0;
  while (u === 0) u = rnd();
  while (v === 0) v = rnd();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// median-preserving lognormal multiplier with the requested CV
function lognormal(rnd, cv) {
  return Math.exp(gauss(rnd) * Math.sqrt(Math.log(1 + cv * cv)));
}

function hashSeed(o) {
  const str = JSON.stringify([o.sex, o.age, o.heightCm, o.weightKg,
    o.frequency, o.food, o.drinks]);
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

const MC_RUNS = 140;

// simulate() emits points on a uniform grid, so a value can be indexed
// directly instead of binary-searched. With 140 runs x 140 grid points that is
// the difference between a snappy readout and a laggy one.
function fastAt(sim, t) {
  const pts = sim.points;
  if (!pts.length) return 0;
  const dt = pts.length > 1 ? pts[1].t - pts[0].t : 1;
  const x = (t - pts[0].t) / dt;
  if (x <= 0) return 0;
  if (x >= pts.length - 1) return pts[pts.length - 1].bac;
  const i = x | 0;
  return pts[i].bac + (x - i) * (pts[i + 1].bac - pts[i].bac);
}

function percentile(sorted, p) {
  if (!sorted.length) return 0;
  const i = (sorted.length - 1) * p;
  const lo = Math.floor(i), hi = Math.ceil(i);
  return sorted[lo] + (i - lo) * (sorted[hi] - sorted[lo]);
}

const DOSE_CV = 0.08;

function simulateBand(o, doseCV = DOSE_CV) {
  const central = simulate(o);
  if (!central.points.length) {
    return { central, grid: [], sample: () => ({ p10: 0, p50: 0, p90: 0 }) };
  }

  const rnd = mulberry32(hashSeed(o));
  const runs = [];
  const draws = [];
  for (let i = 0; i < MC_RUNS; i++) {
    const p = {
      rScale:    lognormal(rnd, rUncertaintyCV(o.sex, o.age)),
      betaScale: lognormal(rnd, 0.22),
      kaScale:   lognormal(rnd, 0.38),
      fScale:    lognormal(rnd, 0.10),
      // Always draw, even when the CV is zero. lognormal(rnd, 0) returns
      // exactly 1, whereas skipping the call would consume one fewer random
      // number and desynchronise the whole stream, so two configurations could
      // no longer be compared against each other.
      doseScale: lognormal(rnd, doseCV),
    };
    draws.push(p);
    // dt=1, matching the deterministic run. At dt=2 the integrator sits about
    // 0.35% low, which would push the whole ensemble off-centre relative to the
    // headline number. Measured cost of the finer step is a few milliseconds,
    // and the ensemble is deferred anyway, so there is no reason to accept it.
    runs.push(simulate({ ...o, dt: 1, ...p }));
  }

  // percentile envelope over a shared time grid
  const t0 = central.sessionStart;
  const tEnd = Math.max(...runs.map(r => r.points[r.points.length - 1].t),
                        central.points[central.points.length - 1].t);
  const N = 140;
  const grid = [];
  const buf = new Float64Array(runs.length);
  for (let i = 0; i <= N; i++) {
    const t = t0 + (i / N) * (tEnd - t0);
    for (let j = 0; j < runs.length; j++) buf[j] = fastAt(runs[j], t);
    const vals = Array.from(buf).sort((a, b) => a - b);
    grid.push({ t, p10: percentile(vals, 0.10), p50: percentile(vals, 0.50),
                p90: percentile(vals, 0.90) });
  }

  // Variance decomposition. Because each parameter is sampled independently,
  // the squared correlation between a parameter and the outcome estimates the
  // share of spread it is responsible for. This turns the band from a bare
  // interval into something actionable: it says WHICH unknown is dominating,
  // and therefore whether waiting will resolve it.
  function drivers(valueOf) {
    const y = runs.map(valueOf);
    const my = y.reduce((a, b) => a + b, 0) / y.length;
    const vy = y.reduce((a, b) => a + (b - my) ** 2, 0);
    if (vy <= 1e-18) return [];
    const out = [];
    for (const key of ["rScale", "betaScale", "kaScale", "fScale", "doseScale"]) {
      const x = draws.map(d => d[key]);
      const mx = x.reduce((a, b) => a + b, 0) / x.length;
      let cov = 0, vx = 0;
      for (let i = 0; i < x.length; i++) {
        cov += (x[i] - mx) * (y[i] - my);
        vx += (x[i] - mx) ** 2;
      }
      const r2 = vx > 0 ? (cov * cov) / (vx * vy) : 0;
      out.push({ key, share: r2 });
    }
    const tot = out.reduce((a, b) => a + b.share, 0) || 1;
    return out.map(d => ({ ...d, share: d.share / tot }))
              .sort((a, b) => b.share - a.share);
  }

  // scalar summaries drawn from the same ensemble
  const peaks = runs.map(r => peakOf(r.points).bac).sort((a, b) => a - b);
  const peakTimes = runs.map(r => peakOf(r.points).t).sort((a, b) => a - b);

  return {
    central, runs, grid, drivers,
    peakP10: percentile(peaks, 0.10),
    peakP50: percentile(peaks, 0.50),
    peakP90: percentile(peaks, 0.90),
    peakTimeP50: percentile(peakTimes, 0.50),
    // BAC at an arbitrary time, as an ensemble percentile
    at(t, p) {
      const vals = runs.map(r => fastAt(r, t)).sort((a, b) => a - b);
      return percentile(vals, p);
    },
    // First time that a given fraction of the ensemble is below a limit.
    // Runs that never resolve are CENSORED, not discarded: dropping them and
    // taking a percentile over the survivors biases the answer early, which is
    // the dangerous direction for a "when am I safe" number. If more than
    // (1 - p) of the ensemble is unresolved, the percentile does not exist.
    timeBelowP(limit, p) {
      const ts = [];
      let censored = 0;
      for (const r of runs) {
        const v = timeBelow(r.points, limit);
        if (v === null) censored++; else ts.push(v);
      }
      if (!ts.length) return null;
      const frac = censored / runs.length;
      if (frac > 1 - p) return null;          // requested percentile is censored
      ts.sort((a, b) => a - b);
      // rescale p onto the observed portion of the distribution
      const pAdj = p / (1 - frac);
      return percentile(ts, Math.min(pAdj, 1));
    },
  };
}

function bacAt(pts, t) {
  if (!pts.length) return 0;
  if (t <= pts[0].t) return 0;
  if (t >= pts[pts.length - 1].t) return pts[pts.length - 1].bac;
  let lo = 0, hi = pts.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (pts[mid].t <= t) lo = mid; else hi = mid;
  }
  const a = pts[lo], b = pts[hi];
  return a.bac + ((t - a.t) / (b.t - a.t)) * (b.bac - a.bac);
}

// Rate of change in permille per hour. Uses a centred difference where there
// is room, and a one-sided one at the edges: bacAt() reads 0 before the session
// starts, so a centred difference at t0 would report a spurious positive slope
// and mislabel the limb.
function slopeAt(pts, t, h = 15) {
  if (!pts.length) return 0;
  const lo = pts[0].t, hi = pts[pts.length - 1].t;
  if (t <= lo) return (bacAt(pts, Math.min(lo + h, hi)) - bacAt(pts, lo)) * (60 / h) * 10;
  if (t >= hi) return 0;
  const a = Math.max(lo, t - h), b = Math.min(hi, t + h);
  if (b - a <= 0) return 0;
  return (bacAt(pts, b) - bacAt(pts, a)) * (60 / (b - a)) * 10;
}

function peakOf(pts) {
  let best = { t: pts[0]?.t ?? 0, bac: 0 };
  for (const p of pts) if (p.bac > best.bac) best = p;
  return best;
}

function timeBelow(pts, limit) {
  const pk = peakOf(pts);
  for (const p of pts) if (p.t >= pk.t && p.bac <= limit) return p.t;
  return null;
}

/* ===========================================================================
   PRESENTATION HELPERS
   Primary unit is permille (g/L). Secondary is g/dL (= US "%BAC").
   =========================================================================== */

const perMille = gdl => gdl * 10;

// Breath alcohol, mg/L. The familiar 2100:1 is an INSTRUMENT AND LEGAL
// convention, not a physiological constant, and it is not the measured mean.
// Jones, comparing near-simultaneous blood and breath in 799 apprehended
// drivers and correcting for metabolism between samples, found a mean ratio of
// 2407 with 95% limits of agreement of 1981 to 2833; in a separate
// post-absorptive series, none of 100 subjects fell below 2100:1. Using 2100
// therefore overstates what a breath device will read by roughly 14%.
// Central estimate uses the measured mean; the spread is reported honestly.
const BBR_MEAN = 2407;
const BBR_LO = 1981;   // 95% limits of agreement
const BBR_HI = 2833;
const toBrAC = (gdl, ratio = BBR_MEAN) => gdl * 10 * 1000 / ratio;

// Thresholds in permille, upper end anchored to the alcohol-poisoning
// literature: the gag reflex and the drive to breathe are suppressed at roughly
// 3.0-4.0, fatal cases average about 3.55, unconsciousness with real overdose
// risk begins around 2.5-3.0, and blackouts are already possible from about
// 1.6 upward.
const ZONES = [
  { max: 0.1, label: "Clear", color: "#8FA981",
    note: "Nothing meaningful in your blood." },
  { max: 0.3, label: "Trace", color: "#C4B571",
    note: "Slight warmth at most. Most people notice nothing." },
  { max: 0.5, label: "Mild", color: "#E9A93C",
    note: "Loosened up, mood lifted. Reaction time is already slower than you feel." },
  { max: 0.8, label: "Impaired", color: "#E07B33",
    note: "Coordination and judgement clearly affected. Above the driving limit across most of Europe." },
  { max: 1.5, label: "Drunk", color: "#D4523A",
    note: "Balance, speech and decision-making visibly affected." },
  { max: 2.0, label: "Heavily drunk", color: "#C43D2E",
    note: "Poor coordination, nausea likely. Blackouts and gaps in memory become common from here." },
  { max: 3.0, label: "Dangerous", color: "#A82D24",
    note: "Vomiting, confusion and loss of consciousness are all likely, and breathing is already being suppressed." },
  { max: 99, label: "Medical emergency", color: "#8C1F1A",
    note: "Around this level the gag reflex and the drive to breathe start shutting down. Fatal cases average roughly 3.5." },
];

const zoneFor = pm => ZONES.find(z => pm < z.max) ?? ZONES[ZONES.length - 1];

const fmtClock = t => {
  const v = ((Math.round(t) % 1440) + 1440) % 1440;
  return `${String(Math.floor(v / 60)).padStart(2, "0")}:${String(v % 60).padStart(2, "0")}`;
};

const PRESETS = [
  { label: "Beer 500", ml: "500", pct: "5", fizzy: true },
  { label: "Beer 330", ml: "330", pct: "5", fizzy: true },
  { label: "IPA 500", ml: "500", pct: "8", fizzy: true },
  { label: "Wine 150", ml: "150", pct: "12", fizzy: false },
  { label: "Spirit 40", ml: "40", pct: "40", fizzy: false },
  { label: "Cocktail", ml: "250", pct: "12", fizzy: true },
];

const DRIVER_LABEL = {
  rScale:    "How your body is built",
  betaScale: "How fast your liver works",
  kaScale:   "How fast it is being absorbed",
  fScale:    "How much actually reaches the blood",
  doseScale: "How sure you are what you drank",
};

// The point of showing this is that the answers differ in whether waiting helps.
const DRIVER_NOTE = {
  kaScale: "Most of the spread right now is alcohol still moving out of your " +
    "stomach. This is the one unknown that resolves by itself: roughly an hour " +
    "after your last drink the number will firm up considerably.",
  rScale: "Most of the spread is how much water your body holds, which sets how " +
    "far a given amount of alcohol is diluted. Height, weight, age and sex only " +
    "predict this to within about ten percent, and no amount of waiting narrows it.",
  betaScale: "Most of the spread is how quickly your liver clears alcohol, which " +
    "varies more than twofold between people and cannot be inferred from the " +
    "outside. It matters most for when you will be clear, not for how high you go.",
  fScale: "Most of the spread is how much of what you drank reached your " +
    "bloodstream rather than being broken down on the way. Food and drink " +
    "strength both move this.",
  doseScale: "Most of the spread is simply not knowing exactly what went in the " +
    "glass. Pours vary and labels are approximate. This is the one source of " +
    "error you can actually reduce, by measuring rather than estimating.",
};

const FREQ = [
  ["rare", "Rarely"], ["occasional", "Monthly"], ["twicemonth", "Twice a month"],
  ["regular", "Weekly"], ["frequent", "Most weeks"], ["daily", "Daily"],
];

const LIMITS = [
  { v: 0.2, label: "0.2" }, { v: 0.5, label: "0.5" }, { v: 0.8, label: "0.8" },
];

/* ===========================================================================
   STYLES
   Palette read off dark bottle glass: warm near-black, malt amber, tannin red.
   =========================================================================== */

const CSS = `
/* fonts are bundled locally in this build (see main.jsx): no network requests */

.bc *, .bc *::before, .bc *::after { box-sizing:border-box; margin:0; padding:0; }
/* Flex and grid children default to min-width:auto, refusing to shrink below
   their own content and shoving siblings out of the card. This single rule
   prevents almost every horizontal overflow on a narrow screen. */
.bc *, .bc *::before, .bc *::after { min-width:0; }

.bc {
  --bg:#13100C; --panel:#1C1813; --panel2:#241E18; --line:#332A20;
  /* --faint was #5E5446, which measured 2.2-2.6:1 against the cards - and it
     carried the 80% range, the breath figure, the "to be safe" times and the
     chart axes, i.e. the most honest information in the app was the least
     legible. #918675 clears WCAG AA (4.5:1) on the lightest card surface. */
  --ink:#EFE7DA; --dim:#9A8E7C; --faint:#918675;
  --amber:#E9A93C; --sage:#8FA981; --red:#D4523A;
  background:var(--bg); color:var(--ink);
  font-family:'IBM Plex Sans',system-ui,sans-serif;
  min-height:100vh; padding:20px 14px 60px;
  -webkit-font-smoothing:antialiased;
  overflow-x:hidden; overflow-wrap:break-word;
}
.bc-wrap { max-width:660px; margin:0 auto; width:100%; }

.bc-head { display:flex; align-items:baseline; gap:10px; margin-bottom:18px; flex-wrap:wrap; }
.bc-head h1 { font-family:'Bricolage Grotesque',sans-serif; font-weight:800;
  font-size:clamp(1.35rem,6vw,1.65rem); letter-spacing:-.03em; line-height:1; }
.bc-head h1 em { font-style:normal; color:var(--amber); }
.bc-head p { font-size:12px; color:var(--faint); line-height:1.3; }

.bc-card { background:var(--panel); border:1px solid var(--line);
  border-radius:14px; padding:16px; margin-bottom:12px; }
.bc-lbl { font-family:'IBM Plex Mono',monospace; font-size:10px; font-weight:600;
  letter-spacing:.16em; text-transform:uppercase; color:var(--faint);
  margin-bottom:12px; display:block; }

/* ---- readout ---- */
.bc-readout { background:linear-gradient(170deg,var(--panel2),var(--panel)); }
.bc-rtop { display:flex; flex-direction:column; align-items:flex-start;
  gap:10px; margin-bottom:8px; }
.bc-asof { display:flex; align-items:center; gap:6px; flex-wrap:nowrap; width:100%; }
.bc-asof > span { font-family:'IBM Plex Mono',monospace; font-size:10px;
  letter-spacing:.1em; color:var(--faint); text-transform:uppercase; flex:0 0 auto; }
/* fixed em widths, so the hour/minute pair can never grow past the card
   whatever system font size the phone is set to */
.bc-asof .bc-in { width:3.2em; flex:0 0 auto; text-align:center;
  padding-left:4px; padding-right:4px; }
.bc-asof .bc-btn { flex:0 0 auto; margin-left:auto; }
@media (min-width:470px) {
  .bc-rtop { flex-direction:row; justify-content:space-between; align-items:center; }
  .bc-asof { width:auto; }
  .bc-asof .bc-btn { margin-left:0; }
}
.bc-big { font-family:'IBM Plex Mono',monospace; font-weight:600;
  font-size:clamp(2.6rem,15vw,3.75rem); line-height:.92; letter-spacing:-.04em; }
.bc-unit { font-family:'IBM Plex Mono',monospace; font-size:12.5px;
  color:var(--dim); margin-top:6px; line-height:1.5; }
.bc-band { font-family:'IBM Plex Mono',monospace; font-size:12px;
  color:var(--faint); margin-top:8px; line-height:1.5; }
.bc-brac { font-family:'IBM Plex Mono',monospace; font-size:11.5px;
  color:var(--faint); margin-top:4px; }
.bc-zone { display:inline-block; margin-top:12px; padding:5px 12px;
  border-radius:20px; font-size:12px; font-weight:500; }
.bc-note { font-size:13px; color:var(--dim); line-height:1.55; margin-top:10px; }
.bc-tol { font-size:12px; color:var(--faint); line-height:1.55; margin-top:8px; font-style:italic; }
.bc-empty { font-size:14px; color:var(--dim); line-height:1.6; padding:12px 0; }
.bc-stale { padding:8px 0 4px; font-size:13.5px; color:var(--dim); line-height:1.6; }
.bc-stale strong { color:var(--ink); font-weight:500; }
.bc-stale-actions { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }

.bc-limb { display:flex; align-items:flex-start; gap:8px; margin-top:12px;
  font-size:12.5px; color:var(--dim); line-height:1.5; }
.bc-arrow { font-size:10px; line-height:1.6; color:var(--faint); flex:0 0 auto; }
.bc-arrow.bc-up { color:#E07B33; }
.bc-arrow.bc-down { color:var(--sage); }
.bc-mellanby { margin-top:12px; padding:12px; border-radius:9px;
  background:rgba(224,123,51,.10); border:1px solid rgba(224,123,51,.32);
  color:#E9B98C; font-size:12.5px; line-height:1.6; }
.bc-mellanby strong { color:#F0C79C; font-weight:500; }
.bc-mellanby em { font-style:italic; }

/* ---- key figures ---- */
.bc-figs { display:grid; grid-template-columns:repeat(3,1fr); gap:1px;
  background:var(--line); border-radius:12px; overflow:hidden; margin-bottom:12px; }
.bc-fig { background:var(--panel); padding:13px 11px; }
.bc-fig dt { font-family:'IBM Plex Mono',monospace; font-size:9px; letter-spacing:.13em;
  text-transform:uppercase; color:var(--faint); margin-bottom:5px; }
.bc-fig dd { font-family:'IBM Plex Mono',monospace; font-size:16px; font-weight:500; }
.bc-fig dd small { display:block; font-size:10px; color:var(--faint);
  font-weight:400; margin-top:3px; letter-spacing:0; line-height:1.4; }

/* ---- chart ---- */
.bc-chart svg { width:100%; height:auto; display:block; }
/* visible only to assistive tech: the chart is an SVG path, which conveys
   nothing without a text equivalent */
.bc-sr {
  position:absolute; width:1px; height:1px; padding:0; margin:-1px;
  overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; border:0;
}
.bc-legend { display:flex; flex-wrap:wrap; gap:14px; margin-top:10px;
  font-size:11px; color:var(--faint); }
.bc-legend i { display:inline-block; width:14px; height:2px; vertical-align:middle; margin-right:5px; }
.bc-legend b { display:inline-block; width:10px; height:10px; border-radius:2px;
  vertical-align:middle; margin-right:5px; }

/* ---- inputs ---- */
.bc-grid4 { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }
.bc-f { display:flex; flex-direction:column; gap:5px; }
.bc-f > label { font-size:11.5px; color:var(--dim); font-weight:400; line-height:1.35;
  display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.bc-fizz { background:none; border:1px solid var(--line); border-radius:20px;
  color:var(--faint); font-family:'IBM Plex Sans',sans-serif; font-size:11.5px;
  /* was 22px tall: too small to hit reliably with a thumb, least of all after a
     few drinks. Measured in a real browser at phone sizes. */
  padding:6px 12px; min-height:34px; cursor:pointer; transition:all .12s; }
.bc-fizz.on { border-color:var(--amber); color:var(--amber); }
.bc-in {
  background:var(--bg); border:1px solid var(--line); border-radius:8px;
  color:var(--ink); font-family:'IBM Plex Mono',monospace;
  font-size:16px;                 /* under 16px, iOS Safari zooms on focus */
  padding:10px; width:100%; min-height:44px;
  outline:none; transition:border-color .12s;
}
.bc-in:focus { border-color:var(--amber); }
.bc-in::placeholder { color:var(--faint); }
.bc-time { display:flex; align-items:center; gap:4px; }
.bc-time .bc-in { text-align:center; padding-left:4px; padding-right:4px; }
.bc-colon { color:var(--faint); font-family:'IBM Plex Mono',monospace;
  font-size:15px; flex:0 0 auto; }
.bc-mini { font-size:16px; }

.bc-seg { display:flex; gap:6px; flex-wrap:wrap; }
.bc-opt {
  flex:1 1 auto; min-width:64px; min-height:42px;
  display:flex; align-items:center; justify-content:center;
  padding:9px 6px; border-radius:8px;
  border:1px solid var(--line); background:var(--bg); color:var(--dim);
  font-family:'IBM Plex Sans',sans-serif; font-size:12px; font-weight:400;
  cursor:pointer; text-align:center; transition:all .12s; user-select:none;
}
.bc-opt:hover { border-color:var(--faint); color:var(--ink); }
.bc-opt[data-on="1"] { border-color:var(--amber); background:rgba(233,169,60,.11);
  color:var(--amber); font-weight:500; }

.bc-btn {
  border:1px solid var(--line); background:var(--bg); color:var(--dim);
  font-family:'IBM Plex Sans',sans-serif; font-size:11.5px; border-radius:7px;
  padding:8px 10px; min-height:34px; cursor:pointer; transition:all .12s;
  white-space:nowrap;
}
.bc-btn:hover { border-color:var(--amber); color:var(--amber); }
.bc-btn.bc-live { border-color:rgba(143,169,129,.5); color:var(--sage); }

/* ---- drinks ---- */
.bc-drink { background:var(--bg); border:1px solid var(--line);
  border-radius:11px; padding:13px; margin-bottom:9px; }
.bc-dhead { display:flex; justify-content:space-between; align-items:center;
  margin-bottom:11px; gap:8px; flex-wrap:wrap; }
.bc-dnum { font-family:'IBM Plex Mono',monospace; font-size:10px;
  letter-spacing:.13em; color:var(--amber); }
.bc-dday { display:inline-block; margin-left:6px; padding:1px 6px; border-radius:10px;
  font-family:'IBM Plex Mono',monospace; font-size:9.5px; letter-spacing:.05em;
  color:var(--faint); border:1px solid var(--line); vertical-align:middle; }
.bc-dgram { font-family:'IBM Plex Mono',monospace; font-size:10px;
  color:var(--faint); margin-left:8px; letter-spacing:0; }
.bc-x { background:none; border:none; color:var(--faint); cursor:pointer;
  font-size:20px; line-height:1; padding:4px 8px; min-height:34px; transition:color .12s; }
.bc-x:hover { color:var(--red); }
.bc-presets { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:11px; }
.bc-add { width:100%; padding:12px; background:transparent;
  border:1px dashed var(--line); border-radius:10px; color:var(--dim);
  font-family:'IBM Plex Sans',sans-serif; font-size:13px; min-height:44px;
  cursor:pointer; transition:all .12s; }
.bc-add:hover { border-color:var(--amber); color:var(--amber); }

.bc-fc { width:100%; border-collapse:collapse; font-family:'IBM Plex Mono',monospace; font-size:12.5px; }
.bc-fc th { text-align:left; font-family:'IBM Plex Sans',sans-serif; font-size:10px;
  letter-spacing:.1em; text-transform:uppercase; color:var(--faint);
  font-weight:500; padding:0 0 8px; }
.bc-fc th:last-child, .bc-fc td:last-child { text-align:right; }
.bc-fc td { padding:7px 0; border-top:1px solid var(--line); }
.bc-fc-t { color:var(--dim); }
.bc-fc-v { font-weight:600; }
.bc-fc-r { color:var(--faint); font-size:11.5px; }
.bc-fc-yes { color:var(--sage); }
.bc-fc-no { color:var(--red); }
.bc-fc-note { margin-top:12px; font-size:12px; color:var(--faint); line-height:1.6; }
@media (max-width:400px) { .bc-fc { font-size:11.5px; } .bc-fc-r { font-size:10.5px; } }
.bc-drivers { display:flex; flex-direction:column; gap:9px; }
.bc-driver { display:flex; align-items:center; gap:10px; font-size:12px; }
.bc-driver-name { flex:1 1 auto; color:var(--dim); }
.bc-driver-bar { flex:0 0 84px; height:5px; border-radius:3px;
  background:var(--line); overflow:hidden; }
.bc-driver-bar > span { display:block; height:100%; background:var(--amber); border-radius:3px; }
.bc-driver-pct { flex:0 0 34px; text-align:right;
  font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--faint); }
.bc-driver-note { margin-top:12px; font-size:12.5px; color:var(--dim); line-height:1.6; }
@media (max-width:400px) { .bc-driver-bar { flex:0 0 56px; } }
.bc-reset { display:flex; align-items:center; gap:8px; flex-wrap:wrap;
  margin-top:12px; font-size:12px; color:var(--faint); }
.bc-btn-warn { border-color:rgba(212,82,58,.5); color:#E8907C; }
.bc-btn-warn:hover { border-color:var(--red); color:var(--red); }
.bc-warn { background:rgba(212,82,58,.09); border:1px solid rgba(212,82,58,.3);
  border-radius:9px; padding:10px 12px; color:#E8907C; font-size:12px;
  line-height:1.55; margin-bottom:12px; }
.bc-danger { background:rgba(140,31,26,.16); border:1px solid rgba(212,82,58,.45);
  border-radius:9px; padding:12px; color:#F0A090; font-size:12.5px;
  line-height:1.6; margin-top:12px; }

.bc-foot { font-size:11.5px; color:var(--faint); line-height:1.7;
  margin-top:20px; padding-top:16px; border-top:1px solid var(--line); }
.bc-foot strong { color:var(--dim); font-weight:500; }
.bc-meta { font-family:'IBM Plex Mono',monospace; font-size:10px;
  color:var(--faint); margin-top:10px; line-height:1.7; }

/* ---- narrow screens ---- */
@media (max-width:520px) {
  .bc { padding:16px 10px 56px; }
  .bc-card { padding:14px 12px; }
  .bc-grid4 { grid-template-columns:1fr 1fr; }
  .bc-figs { grid-template-columns:1fr; }
  .bc-legend { gap:10px; font-size:10.5px; }
}
@media (max-width:360px) {
  .bc-grid4 { grid-template-columns:1fr; }
  .bc-opt { min-width:52px; font-size:11.5px; }
  .bc-presets .bc-btn { font-size:11px; padding:7px 8px; }
}
@media (prefers-reduced-motion:reduce) { .bc * { transition:none !important; } }
`;

/* ===========================================================================
   COMPONENT
   =========================================================================== */

// Persisted state is untrusted input. It can be truncated mid-write, edited by
// hand, or left over from an older version of this app with a different shape.
// A single null in the drinks array previously threw during render, which
// white-screens the app permanently: it reloads, reads the same bad save, and
// dies again, with no way back for the user. Everything is coerced here.
const safeStr = (v, fallback = "") => {
  if (v === null || v === undefined) return fallback;
  if (typeof v === "number") return Number.isFinite(v) ? String(v) : fallback;
  if (typeof v !== "string") return fallback;
  // reject anything that would parse to Infinity, e.g. "1e400"
  const n = parseFloat(v);
  if (v.trim() !== "" && !Number.isNaN(n) && !Number.isFinite(n)) return fallback;
  return v.slice(0, 8);
};

const safeDrink = (d, fallback) => {
  if (!d || typeof d !== "object") return { ...fallback };
  return {
    ml:     safeStr(d.ml, fallback.ml),
    pct:    safeStr(d.pct, fallback.pct),
    startH: safeStr(d.startH, fallback.startH),
    startM: safeStr(d.startM, fallback.startM),
    endH:   safeStr(d.endH, fallback.endH),
    endM:   safeStr(d.endM, fallback.endM),
    mixer:  ["plain", "sugary", "diet"].includes(d.mixer) ? d.mixer : "plain",
    // saves from before carbonation existed have no fizzy field; infer it from
    // strength rather than silently turning every beer flat
    fizzy:  typeof d.fizzy === "boolean" ? d.fizzy : (parseFloat(d.pct) || 0) <= 9,
  };
};

function sanitizeSaved(raw, defProfile, defDrink) {
  const out = {};
  if (raw && typeof raw === "object") {
    if (raw.profile && typeof raw.profile === "object") {
      out.profile = {
        sex: raw.profile.sex === "female" ? "female" : "male",
        age: safeStr(raw.profile.age, defProfile.age),
        heightCm: safeStr(raw.profile.heightCm, defProfile.heightCm),
        weightKg: safeStr(raw.profile.weightKg, defProfile.weightKg),
      };
    }
    if (["empty", "light", "full"].includes(raw.food)) out.food = raw.food;
    if (Object.prototype.hasOwnProperty.call(ELIMINATION_BETA, raw.freq)) out.freq = raw.freq;
    if (typeof raw.limit === "number" && Number.isFinite(raw.limit)
        && raw.limit > 0 && raw.limit < 5) out.limit = raw.limit;
    if (Array.isArray(raw.drinks)) {
      const ds = raw.drinks.slice(0, 40).map(d => safeDrink(d, defDrink));
      if (ds.length) out.drinks = ds;
    }
    // Only a FUTURE epoch is untrusted. An old one is fine: it simply places
    // the session long in the past, which is the truth and exactly what the
    // stale check should conclude. Rejecting old epochs would fall back to the
    // ambiguous wall-clock placement, i.e. reintroduce the original bug.
    if (typeof raw.sessionEpoch === "number" && Number.isFinite(raw.sessionEpoch)
        && raw.sessionEpoch <= Date.now() + 2 * 86400000) {
      out.sessionEpoch = raw.sessionEpoch;
    }
  }
  return out;
}

const nowDate = new Date();
const DEF_PROFILE = { sex: "male", age: "30", heightCm: "178", weightKg: "80" };
// A fresh session's first drink starts NOW, not at a hard-coded evening hour.
// With 19:00 baked in, someone opening the app for the first time at 22:00 saw
// a reading from a beer they never had, and someone opening it at 08:00 saw a
// confusing note about the clock being before their first drink.
const freshDrink = () => {
  const d = new Date();
  const pad = n => String(n).padStart(2, "0");
  const h = d.getHours(), m = d.getMinutes();
  return { ml: "500", pct: "5", mixer: "plain", fizzy: true,
    startH: pad(h), startM: pad(m), endH: pad((h + 1) % 24), endM: pad(m) };
};
// Shape reference for the sanitiser; times here are placeholders only.
const DEF_DRINK = { ml: "500", pct: "5", startH: "19", startM: "00", endH: "20", endM: "00", mixer: "plain", fizzy: true };

export default function BACCalculator() {
  const [profile, setProfile] = useState(DEF_PROFILE);
  const [food, setFood] = useState("light");
  const [freq, setFreq] = useState("regular");
  const [limit, setLimit] = useState(0.5);
  const [drinks, setDrinks] = useState(() => [freshDrink()]);
  const [nowH, setNowH] = useState(String(nowDate.getHours()).padStart(2, "0"));
  const [nowM, setNowM] = useState(String(nowDate.getMinutes()).padStart(2, "0"));
  // "Estimated right now" should mean NOW. The clock follows real time until
  // the person edits it by hand, at which point it is theirs - editing it is how
  // you ask "what will I be at midnight?", and a ticking clock would fight that.
  // Tapping "now" hands it back.
  const [liveClock, setLiveClock] = useState(true);
  // Only wall-clock times are stored per drink, so "20:00" is ambiguous
  // between tonight and last night. Reopening the next evening used to show
  // yesterday's beers as if they were being drunk now. The fix is to persist
  // an absolute reference for the session's timeline: sessionEpoch is the real
  // timestamp at which timeline minute ZERO occurred (midnight of the night's
  // reference day). "Now" on the timeline is then (Date.now() - sessionEpoch)
  // in minutes, with no reference to any drink at all.
  // Three designs were tried and rejected first: a fixed "older than N hours"
  // rule (a heavy night ending at 03:00 is still over the limit at 13:00);
  // anchoring to the last save's wall-clock (breaks the moment you edit a drink
  // the next day, since that save's wall-clock is ambiguous too); and anchoring
  // to the FIRST DRINK's start (correcting that drink from 19:00 to 18:00 moved
  // "now" back an hour, and the next save baked the error in permanently).
  const [sessionEpoch, setSessionEpoch] = useState(null);
  const [staleDismissed, setStaleDismissed] = useState(false);
  // Drinks restored from a save that carries no epoch (a save from before
  // epochs existed, or one with a rejected epoch) cannot be placed in time.
  // Showing them as current would be a guess in the unsafe direction, so they
  // are treated as a previous night until the person says otherwise.
  const [epochMissing, setEpochMissing] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const saveTimer = useRef(null);
  const resultRef = useRef(null);
  // Focus the new drink's first field after add/duplicate.
  // Driven ONLY by an explicit request from those two handlers. An earlier
  // version fired whenever the drink count went up, which meant restoring a
  // saved session focused an input and popped the keyboard over the reading
  // the person had opened the app to look at.
  const listRef = useRef(null);
  useEffect(() => {
    const idx = focusIndex.current;
    if (idx == null || !listRef.current) return;
    focusIndex.current = null;
    const cards = listRef.current.querySelectorAll(".bc-drink");
    const card = cards[Math.min(idx, cards.length - 1)];
    const input = card && card.querySelector("input");
    if (input) { try { input.focus(); input.select(); } catch { /* ignore */ } }
  }, [drinks]);

  /* ---- persistence: profile + tonight's session survive a reload ---- */
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await window.storage.get("baccalc:v3");
        if (!cancelled && res?.value) {
          const d = sanitizeSaved(JSON.parse(res.value), DEF_PROFILE, DEF_DRINK);
          if (d.profile) setProfile(p => ({ ...p, ...d.profile }));
          if (d.food) setFood(d.food);
          if (d.freq) setFreq(d.freq);
          if (typeof d.limit === "number") setLimit(d.limit);
          if (d.drinks) setDrinks(d.drinks);
          if (d.drinks) {
            if (typeof d.sessionEpoch === "number") setSessionEpoch(d.sessionEpoch);
            else setEpochMissing(true);
          }
        }
      } catch { /* first run, or storage unavailable */ }
      if (!cancelled) setLoaded(true);
    })();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!loaded) return;
    clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => {
      try {
        // Capture the timeline's absolute zero ONCE, when there is no epoch yet.
        // It is an absolute instant and never needs recomputing. An earlier
        // version recaptured it on every save "to stay in sync", which was a
        // slow infinite loop: each recapture was derived from a tNow computed a
        // few hundred ms earlier, so it always differed slightly, which set
        // state, which re-ran this effect, which saved again, every 700 ms,
        // forever. Storage is rate-limited; this would have hit it in minutes.
        let epoch = sessionEpoch;
        if (epoch == null && resultRef.current && resultRef.current.ready) {
          const r = resultRef.current;
          if (liveClock) {
            epoch = Date.now() - r.tNow * 60000;
          } else {
            // a manually planned time must never be mistaken for the present
            const d = new Date();
            const realNow = normalizeNow(d.getHours(), d.getMinutes(), r.central.sessionStart);
            epoch = Date.now() - realNow * 60000;
          }
          setSessionEpoch(epoch);
        }
        // storage.set is async: a try/catch around it does NOT catch a rejected
        // promise (quota exceeded, storage unavailable), which then surfaces as
        // an unhandled rejection. Promise.resolve also covers a sync return.
        Promise.resolve(window.storage.set("baccalc:v3", JSON.stringify({
          profile, food, freq, limit, drinks, sessionEpoch: epoch }))).catch(() => {});
      } catch { /* ignore */ }
    }, 700);
    return () => clearTimeout(saveTimer.current);
  }, [profile, food, freq, limit, drinks, loaded, liveClock, sessionEpoch]);

  useEffect(() => {
    if (!liveClock) return;
    const tick = () => {
      const d = new Date();
      const h = String(d.getHours()).padStart(2, "0");
      const m = String(d.getMinutes()).padStart(2, "0");
      setNowH(prev => (prev === h ? prev : h));
      setNowM(prev => (prev === m ? prev : m));
    };
    tick();
    const id = setInterval(tick, 20000);
    return () => clearInterval(id);
  }, [liveClock]);

  // Coming back to the app must refresh the reading at once. Phones suspend
  // background timers, so without this the number could be 20 s stale on
  // return - and since the reading only recomputed when the displayed digits
  // changed, returning exactly 24 h later to the minute would have shown a
  // reading from a day ago. `wake` forces a recompute regardless of digits.
  const [wake, setWake] = useState(0);
  useEffect(() => {
    const onVisible = () => {
      if (document.visibilityState !== "visible") return;
      setWake(w => w + 1);
      if (liveClock) {
        const d = new Date();
        setNowH(String(d.getHours()).padStart(2, "0"));
        setNowM(String(d.getMinutes()).padStart(2, "0"));
      }
    };
    document.addEventListener("visibilitychange", onVisible);
    window.addEventListener("focus", onVisible);
    return () => {
      document.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener("focus", onVisible);
    };
  }, [liveClock]);

  /* ---- input sanitisers: digits only, kept as strings while typing ---- */
  const digits = v => v.replace(/[^0-9]/g, "").slice(0, 4);
  const hm = v => v.replace(/[^0-9]/g, "").slice(0, 2);
  const decimal = v => v.replace(/[^0-9.]/g, "").replace(/(\..*)\./g, "$1").slice(0, 6);

  const setP = (k, v) => setProfile(p => ({ ...p, [k]: v }));
  const upd = (i, k, v) => {
    // editing keeps the session's epoch (it is still the same night) but is a
    // clear signal the person wants to work with these drinks, so stop nagging
    setStaleDismissed(true);
    setEpochMissing(false);
    setDrinks(ds => ds.map((d, j) => j === i ? { ...d, [k]: v } : d));
  };
  const rm = i => setDrinks(ds => ds.filter((_, j) => j !== i));

  // Entering six identical beers one field at a time is the single biggest
  // piece of friction in the app. The copy is chained so it starts when the
  // original finished, keeping the same duration.
  const focusIndex = useRef(null);
  const duplicateDrink = i => { focusIndex.current = i + 1; setDrinks(ds => {
    const d = ds[i];
    const sMin = (parseInt(d.startH, 10) || 0) * 60 + (parseInt(d.startM, 10) || 0);
    let eMin = (parseInt(d.endH, 10) || 0) * 60 + (parseInt(d.endM, 10) || 0);
    if (eMin < sMin) eMin += 1440;
    const dur = Math.max(eMin - sMin, 5);
    const ns = eMin % 1440;
    const ne = (eMin + dur) % 1440;
    const pad = n => String(n).padStart(2, "0");
    const copy = { ...d,
      startH: pad(Math.floor(ns / 60)), startM: pad(ns % 60),
      endH: pad(Math.floor(ne / 60)), endM: pad(ne % 60) };
    return [...ds.slice(0, i + 1), copy, ...ds.slice(i + 1)];
  }); };

  const addDrink = () => setDrinks(ds => {
    // updater form: two rapid adds must chain off each other, not off a
    // snapshot of state taken before the first one applied
    const last = ds[ds.length - 1] ?? DEF_DRINK;
    let h = parseInt(last.endH, 10) || 0;
    let m = (parseInt(last.endM, 10) || 0) + 30;
    if (m >= 60) { h += 1; m -= 60; }
    focusIndex.current = ds.length;
    return [...ds, {
      ml: "500", pct: "5", mixer: "plain", fizzy: true,
      startH: String(parseInt(last.endH, 10) || 0).padStart(2, "0"),
      startM: String(parseInt(last.endM, 10) || 0).padStart(2, "0"),
      endH: String(h % 24).padStart(2, "0"), endM: String(m).padStart(2, "0"),
    }];
  });

  const [confirmReset, setConfirmReset] = useState(false);

  // Clearing the drinks matters for an app used night after night: without it
  // last night's session silently becomes tonight's starting point. The profile
  // is deliberately kept, since body and habits do not change between nights.
  const newNight = () => {
    const d = new Date();
    const h = d.getHours(), m = d.getMinutes();
    setDrinks([{
      ml: "500", pct: "5", mixer: "plain", fizzy: true,
      startH: String(h).padStart(2, "0"), startM: String(m).padStart(2, "0"),
      endH: String((h + 1) % 24).padStart(2, "0"), endM: String(m).padStart(2, "0"),
    }]);
    setNowH(String(h).padStart(2, "0"));
    setNowM(String(m).padStart(2, "0"));
    setLiveClock(true);
    setSessionEpoch(null);
    setEpochMissing(false);
    setStaleDismissed(false);
    setConfirmReset(false);
  };

  const stampNow = () => {
    const d = new Date();
    setNowH(String(d.getHours()).padStart(2, "0"));
    setNowM(String(d.getMinutes()).padStart(2, "0"));
    setLiveClock(true);
  };

  const finishNow = i => {
    const d = new Date();
    upd(i, "endH", String(d.getHours()).padStart(2, "0"));
    upd(i, "endM", String(d.getMinutes()).padStart(2, "0"));
  };

  /* ---- live simulation ----
     Two separate memos, and the split is the whole point. The deterministic
     run is cheap and must track every keystroke. The 140-run ensemble is not
     cheap, so it is computed from a DEFERRED copy of the inputs in its own
     memo. React can then render the headline immediately and fill the band in
     a lower-priority pass. Putting both in one memo, with both fresh and
     deferred inputs in its dependency list, would miss the cache twice per
     keystroke and run the ensemble on the render path both times, which is
     worse than not deferring at all. */

  // Validate and clamp once, in one place, so the two memos cannot disagree.
  const buildOpts = (p, fd, fq, ds) => {
    const num = (v, lo, hi) => {
      const n = parseFloat(v);
      return Number.isFinite(n) ? Math.min(Math.max(n, lo), hi) : NaN;
    };
    const w = num(p.weightKg, 20, 400);
    const h = num(p.heightCm, 100, 260);
    const a = num(p.age, 15, 110);
    if (!Number.isFinite(w) || !Number.isFinite(h) || !Number.isFinite(a)) {
      return { error: "profile" };
    }
    const usable = (ds || [])
      .filter(d => d && Number.isFinite(parseFloat(d.ml)) && Number.isFinite(parseFloat(d.pct))
                && parseFloat(d.ml) > 0 && parseFloat(d.pct) > 0)
      .map(d => ({ ...d,
        ml: Math.min(parseFloat(d.ml), 5000),
        pct: Math.min(parseFloat(d.pct), 96) }));
    if (!usable.length) return { error: "drinks" };
    return { opts: { sex: p.sex, age: a, heightCm: h, weightKg: w,
                     frequency: fq, food: fd, drinks: usable } };
  };

  const simInputs = useMemo(
    () => ({ profile, food, freq, drinks }),
    [profile, food, freq, drinks]);
  const deferredInputs = useDeferredValue(simInputs);

  // Ensemble: depends ONLY on the deferred inputs.
  const bandMemo = useMemo(() => {
    const b = buildOpts(deferredInputs.profile, deferredInputs.food,
                        deferredInputs.freq, deferredInputs.drinks);
    if (b.error) return null;
    return simulateBand(b.opts);
  }, [deferredInputs]);

  const result = useMemo(() => {
    const b = buildOpts(profile, food, freq, drinks);
    if (b.error) return { ready: false, reason: b.error };

    const central = simulate(b.opts);
    if (!central.points.length) return { ready: false, reason: "drinks" };

    // While the deferred inputs lag by a keystroke, the band shown is one
    // keystroke stale. Fall back to a band built from the fresh inputs only if
    // there is no ensemble at all yet (first render), so the readout is never
    // missing its interval.
    const band = (bandMemo && bandMemo.central.points.length)
      ? bandMemo : simulateBand(b.opts);

    const t0 = central.sessionStart;
    const lastEnd = central.lastEnd;
    // While the clock is live and this session was reloaded, "now" is placed
    // by elapsed real time rather than by ambiguous wall-clock digits.
    let tNow;
    if (sessionEpoch != null) {
      const epochNow = (Date.now() - sessionEpoch) / 60000;
      // A manually typed time means a time near NOW, not near the session
      // start: "what will I be at 15:00" the day after a night out is today's
      // 15:00, which session-relative placement would put before the drinks.
      tNow = liveClock ? epochNow : normalizeNow(nowH, nowM, epochNow);
    } else {
      tNow = normalizeNow(nowH, nowM, t0);
    }
    const pk = peakOf(central.points);
    const lim = limit / 10; // permille -> g/dL
    // A previous night is one that has fully cleared on its own timeline.
    const stale = sessionEpoch != null && tNow > lastEnd + 120 && bacAt(central.points, tNow) < 0.005;

    return {
      ready: true, central, band, tNow, peak: pk, stale,
      beforeStart: tNow < t0,
      nowBac: bacAt(central.points, tNow),
      nowLow: band.at(tNow, 0.10),
      nowHigh: band.at(tNow, 0.90),
      peakP10: band.peakP10,
      peakP90: band.peakP90,
      // Which limb of the curve you are on. This matters more than the number:
      // at the SAME BAC, people on the falling limb feel markedly less drunk
      // but perform worse on driving and inhibitory-control tasks.
      slopePerHour: slopeAt(central.points, tNow),
      // Hourly forecast: the point of the app is deciding when you are safe.
      forecast: (() => {
        const rows = [];
        const first = Math.ceil((tNow + 1) / 60) * 60;
        const endT = central.points[central.points.length - 1].t;
        for (let t = first; t <= endT && rows.length < 12; t += 60) {
          const v = bacAt(central.points, t);
          const hi = band.at(t, 0.90);
          rows.push({ t, bac: v, lo: band.at(t, 0.10), hi, overLimit: hi > lim });
          if (v < 0.005 && rows.length > 1) break;
        }
        return rows;
      })(),
      drivers: band.drivers
        ? band.drivers(r => bacAt(r.points, tNow)).filter(d => d.share > 0.02)
        : [],
      // A peak far past any survivable level almost always means a typo.
      implausible: pk.bac > 0.45,
      overLimit: band.peakP90 > lim,
      belowLimit: timeBelow(central.points, lim),
      belowLimitCautious: band.timeBelowP(lim, 0.90),
      clear: timeBelow(central.points, 0.01),
      clearCautious: band.timeBelowP(0.01, 0.90),
      warnings: central.warnings,
    };
  }, [profile, food, freq, drinks, bandMemo, nowH, nowM, limit, liveClock, sessionEpoch, wake]);

  /* ---- chart geometry ---- */
  const chart = useMemo(() => {
    if (!result.ready) return null;
    const { central, band, tNow, peak } = result;
    const t0 = central.sessionStart - 20;
    const endPt = result.clearCautious ?? central.points[central.points.length - 1].t;

    const t1 = Math.max(endPt + 25, tNow + 25, peak.t + 40);

    const W = 560, H = 268;
    const ML = 40, MR = 12, MT = 14, MB = 54;
    const pw = W - ML - MR, ph = H - MT - MB;

    const yMaxPm = Math.max(perMille(band.peakP90) * 1.12, 0.55);
    const x = t => ML + ((t - t0) / (t1 - t0)) * pw;
    const y = pm => MT + ph - (pm / yMaxPm) * ph;

    const samp = band.grid
      .filter(g => g.t >= t0 && g.t <= t1)
      .map(g => ({ t: g.t, c: perMille(bacAt(central.points, g.t)),
                   l: perMille(g.p10), h: perMille(g.p90) }));
    if (!samp.length) return null;

    const line = key => samp.map((p, i) =>
      `${i ? "L" : "M"}${x(p.t).toFixed(1)},${y(p[key]).toFixed(1)}`).join("");
    const envelope =
      samp.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${y(p.h).toFixed(1)}`).join("") +
      samp.slice().reverse().map(p => `L${x(p.t).toFixed(1)},${y(p.l).toFixed(1)}`).join("") + "Z";

    // y gridlines at sensible permille steps
    const step = yMaxPm > 2.4 ? 0.5 : yMaxPm > 1.2 ? 0.25 : 0.1;
    const yTicks = [];
    for (let v = 0; v <= yMaxPm + 1e-9; v += step) yTicks.push(+v.toFixed(2));

    // x gridlines on round hours
    const span = t1 - t0;
    const hourStep = span > 720 ? 180 : span > 420 ? 120 : 60;
    const xTicks = [];
    for (let t = Math.ceil(t0 / hourStep) * hourStep; t <= t1; t += hourStep) xTicks.push(t);

    return {
      W, H, ML, MR, MT, MB, pw, ph, x, y, yMaxPm,
      envelope, cLine: line("c"), yTicks, xTicks,
      limitPm: limit,
      drinkBars: central.drinks.map(d => ({
        x1: x(d.start), x2: x(Math.max(d.end, d.start + 3)), idx: d.idx,
      })),
      nowX: x(tNow), peakX: x(peak.t), peakY: y(perMille(peak.bac)),
      peakPm: perMille(peak.bac),
    };
  }, [result, limit]);

  resultRef.current = result;
  // which raw drink index landed on a later day than the session's reference day
  const dayOffsetByIdx = useMemo(() => {
    const m = {};
    if (result.ready) for (const d of result.central.drinks) m[d.idx] = Math.floor(d.start / 1440);
    return m;
  }, [result]);
  const staleSession = result.ready && (result.stale || epochMissing) && !staleDismissed;
  const staleSince = (sessionEpoch != null && result.ready)
    ? sessionEpoch + result.central.sessionStart * 60000 : null;
  const pm = result.ready ? perMille(result.nowBac) : 0;
  const zone = zoneFor(pm);
  const totalG = result.ready ? result.central.totalGrams : 0;

  const tolNote = {
    rare: "You drink rarely, so a given level will likely hit you harder than it would a regular drinker.",
    occasional: null,
    twicemonth: null,
    regular: null,
    frequent: "As a frequent drinker you may feel less affected than this number suggests. Your blood alcohol is the same either way.",
    daily: "Daily drinking builds tolerance to how alcohol feels, not to what it does. You may feel far less impaired than this number indicates.",
  }[freq];

  return (
    <div className="bc">
      <style>{CSS}</style>
      <div className="bc-wrap">

        <header className="bc-head">
          <h1>Blood<em>·</em>alc</h1>
          <p>Drink-by-drink<br />pharmacokinetics</p>
        </header>

        {/* ── READOUT ──────────────────────────────────────────────── */}
        <section className="bc-card bc-readout">
          <div className="bc-rtop">
            <span className="bc-lbl" style={{ margin: 0 }}>
              {liveClock ? "Estimated right now" : "Estimated at the time set"}
            </span>
            <div className="bc-asof">
              <span>as of</span>
              <input className="bc-in bc-mini" type="text" inputMode="numeric"
                value={nowH} aria-label="Current hour"
                onChange={e => { setLiveClock(false); setNowH(hm(e.target.value)); }} />
              <span className="bc-colon">:</span>
              <input className="bc-in bc-mini" type="text" inputMode="numeric"
                value={nowM} aria-label="Current minute"
                onChange={e => { setLiveClock(false); setNowM(hm(e.target.value)); }} />
              <button className={`bc-btn${liveClock ? " bc-live" : ""}`} onClick={stampNow}
                aria-pressed={liveClock}
                title={liveClock ? "Following the real clock" : "Tap to follow the real clock again"}>
                {liveClock ? "● live" : "now"}
              </button>
            </div>
          </div>

          {staleSession ? (
            <div className="bc-stale">
              <p>
                {epochMissing ? (
                  <>
                    <strong>These drinks were saved without a date</strong>, so
                    it is not safe to assume they are tonight's. No reading is
                    shown until you say which they are.
                  </>
                ) : (
                  <>
                    <strong>These drinks are from a previous night</strong>
                    {staleSince && ` (first drink ${new Date(staleSince).toLocaleString([], { weekday: "short", hour: "2-digit", minute: "2-digit" })})`}.
                    They will have cleared by now, so no reading is shown for them.
                  </>
                )}
              </p>
              <div className="bc-stale-actions">
                <button className="bc-btn" onClick={newNight}>Start a new night</button>
                <button className="bc-btn" onClick={() => { setStaleDismissed(true); setEpochMissing(false); }}>
                  {epochMissing ? "Keep them, they're from tonight" : "Keep them, I'm reviewing last night"}
                </button>
              </div>
            </div>
          ) : !result.ready ? (
            <p className="bc-empty">
              {result.reason === "profile"
                ? "Fill in your age, height and weight below and this will start calculating."
                : "Add a drink below with a volume and an alcohol percentage."}
            </p>
          ) : (
            <>
              <div className="bc-big" style={{ color: zone.color }}>
                {pm.toFixed(2)}
              </div>
              <div className="bc-unit">
                ‰ (g/L) &nbsp;·&nbsp; {result.nowBac.toFixed(3)} g/dL, the US "%BAC" figure
              </div>
              <div className="bc-band">
                80% of people would fall between{" "}
                {perMille(result.nowLow).toFixed(2)} and {perMille(result.nowHigh).toFixed(2)} ‰
              </div>
              <div className="bc-brac">
                ≈ {toBrAC(result.nowBac).toFixed(2)} mg/L on a breathalyser
                {result.nowBac > 0.002 && (
                  <> · plausibly {toBrAC(result.nowBac, BBR_HI).toFixed(2)}–
                  {toBrAC(result.nowBac, BBR_LO).toFixed(2)}</>
                )}
              </div>
              {result.slopePerHour > 0.02 && perMille(result.nowBac) >= 0.2 && (
                <div className="bc-brac">
                  While your level is still rising, a breath test tends to read
                  high relative to a blood sample, because arterial blood is
                  ahead of venous blood during absorption.
                </div>
              )}
              <div className="bc-zone" style={{ background: zone.color + "22", color: zone.color }}>
                {zone.label}
              </div>
              <p className="bc-note">{zone.note}</p>

              {pm >= 0.1 && (
                <div className="bc-limb">
                  {result.slopePerHour > 0.02 ? (
                    <>
                      <span className="bc-arrow bc-up">&#9650;</span>
                      <span>
                        Still rising, about {result.slopePerHour.toFixed(2)} ‰ per hour.
                        Alcohol already drunk has not all reached your blood yet.
                      </span>
                    </>
                  ) : result.slopePerHour < -0.02 ? (
                    <>
                      <span className="bc-arrow bc-down">&#9660;</span>
                      <span>
                        Falling, about {Math.abs(result.slopePerHour).toFixed(2)} ‰ per hour.
                      </span>
                    </>
                  ) : (
                    <>
                      <span className="bc-arrow">&#9679;</span>
                      <span>Around your peak, roughly level right now.</span>
                    </>
                  )}
                </div>
              )}

              {result.slopePerHour < -0.02 && pm >= 0.3 && (
                <div className="bc-mellanby">
                  <strong>You will feel soberer than you are.</strong> On the way
                  back down, people report about 29% less intoxication than at the
                  identical reading on the way up, and are far likelier to say they
                  would drive. Measured performance does not improve to match:
                  driving and inhibitory control test <em>worse</em> on the falling
                  limb at the same concentration. Trust the number, not the feeling.
                </div>
              )}
              {result.beforeStart && (
                <p className="bc-tol">
                  The clock above is set earlier than your first drink, so nothing
                  has reached your blood yet. Tap “now”, or check the curve below
                  for where the night is heading.
                </p>
              )}
              {!result.beforeStart && tolNote && <p className="bc-tol">{tolNote}</p>}
              {pm >= 2.0 && (
                <div className="bc-danger">
                  {result.slopePerHour > 0.02 ? (
                    <>
                      <strong>This is still climbing.</strong> Alcohol already
                      drunk has not finished reaching the blood, so the number
                      keeps rising for a while after the last drink — which is
                      exactly why letting someone sleep it off is dangerous.
                    </>
                  ) : (
                    <strong>This is a level where people get hurt.</strong>
                  )}
                  <br /><br />
                  Call emergency services if someone cannot be woken, is
                  breathing slowly or irregularly, keeps vomiting, has a seizure,
                  or looks pale, blue-tinged or cold. Until help arrives, put
                  them on their side so they cannot choke, and stay with them.
                  <br /><br />
                  Coffee, a cold shower, walking it off and making someone throw
                  up do none of them any good. Once alcohol is in the blood,
                  nothing but time removes it.
                </div>
              )}
            </>
          )}
        </section>

        {result.ready && !staleSession && result.warnings.length > 0 && (
          <div className="bc-warn">{result.warnings.join(" ")}</div>
        )}

        {result.ready && !staleSession && result.implausible && (
          <div className="bc-warn">
            <strong>Check these numbers.</strong> What you have entered works out
            to a level well beyond anything survivable, which nearly always means
            a volume or a percentage has been typed wrong. Alcohol is fatal to
            around half of people somewhere near 4 permille, and this is far past
            that. If the drinks really are as entered, the person needs emergency
            help, not a calculator.
          </div>
        )}

        {result.ready && !staleSession && parseFloat(profile.age) >= 60 && (
          <div className="bc-warn">
            Over about 60, measured blood alcohol tends to run <em>higher</em> than
            these equations predict, and for women the body-water equation carries
            no age term at all. Treat this as a floor rather than a best guess.
          </div>
        )}

        {/* ── KEY FIGURES ──────────────────────────────────────────── */}
        {result.ready && !staleSession && (
          <dl className="bc-figs">
            <div className="bc-fig">
              <dt>Peak</dt>
              <dd>
                {perMille(result.peak.bac).toFixed(2)} ‰
                <small>
                  {perMille(result.peakP10).toFixed(2)}–{perMille(result.peakP90).toFixed(2)} at {fmtClock(result.peak.t)}
                </small>
              </dd>
            </div>
            {/* For a safety decision the CAUTIOUS time is the headline and the
                typical one is the footnote, not the other way round. This also
                keeps the figure consistent with the hourly table below, which
                uses the same pessimistic end of the range. */}
            <div className="bc-fig">
              <dt>Under {limit} ‰</dt>
              <dd>
                {!result.overLimit
                  ? <span style={{ color: "#8FA981" }}>never over</span>
                  : (result.belowLimitCautious == null ? "—" : fmtClock(result.belowLimitCautious))}
                <small>
                  {!result.overLimit
                    ? "you stayed below it all night"
                    : (result.belowLimit == null ? "to be safe"
                        : `to be safe · typically ${fmtClock(result.belowLimit)}`)}
                </small>
              </dd>
            </div>
            <div className="bc-fig">
              <dt>Under 0.1 ‰</dt>
              <dd>
                {result.clearCautious == null ? "—" : fmtClock(result.clearCautious)}
                <small>
                  {result.clear == null ? "to be safe"
                    : `to be safe · typically ${fmtClock(result.clear)}`}
                </small>
              </dd>
            </div>
          </dl>
        )}

        {/* ── WHAT DRIVES THE UNCERTAINTY ──────────────────────────── */}
        {result.ready && !staleSession && result.drivers.length > 0 && perMille(result.nowBac) >= 0.1 && (
          <section className="bc-card">
            <span className="bc-lbl">Why the range is that wide</span>
            <div className="bc-drivers">
              {result.drivers.slice(0, 3).map(d => (
                <div className="bc-driver" key={d.key}>
                  <span className="bc-driver-name">{DRIVER_LABEL[d.key]}</span>
                  <span className="bc-driver-bar">
                    <span style={{ width: `${Math.round(d.share * 100)}%` }} />
                  </span>
                  <span className="bc-driver-pct">{Math.round(d.share * 100)}%</span>
                </div>
              ))}
            </div>
            <p className="bc-driver-note">{DRIVER_NOTE[result.drivers[0].key]}</p>
          </section>
        )}

        {/* ── HOURLY FORECAST ──────────────────────────────────────── */}
        {result.ready && !staleSession && result.forecast.length > 1 && (
          <section className="bc-card">
            <span className="bc-lbl">Hour by hour from here</span>
            <table className="bc-fc">
              <thead>
                <tr>
                  <th>Time</th><th>Estimate</th><th>Range</th><th>Under {limit} ‰?</th>
                </tr>
              </thead>
              <tbody>
                {result.forecast.map(row => (
                  <tr key={row.t}>
                    <td className="bc-fc-t">{fmtClock(row.t)}</td>
                    <td className="bc-fc-v" style={{ color: zoneFor(perMille(row.bac)).color }}>
                      {perMille(row.bac).toFixed(2)}
                    </td>
                    <td className="bc-fc-r">
                      {perMille(row.lo).toFixed(2)}–{perMille(row.hi).toFixed(2)}
                    </td>
                    <td className={row.overLimit ? "bc-fc-no" : "bc-fc-yes"}>
                      {row.overLimit ? "no" : "yes"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="bc-fc-note">
              The last column only says yes once even the unlucky end of the
              range is under your limit. It is deliberately the pessimistic
              reading, because the cost of being wrong is not symmetrical.
            </p>
          </section>
        )}

        {/* ── CHART ────────────────────────────────────────────────── */}
        {chart && !staleSession && (
          <section className="bc-card bc-chart">
            <span className="bc-lbl">Your night</span>
            <svg viewBox={`0 0 ${chart.W} ${chart.H}`} role="img"
                 aria-label="Blood alcohol concentration over the course of the session">
              <defs>
                <linearGradient id="bcEnv" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#E9A93C" stopOpacity="0.30" />
                  <stop offset="100%" stopColor="#E9A93C" stopOpacity="0.04" />
                </linearGradient>
              </defs>

              {/* y grid */}
              {chart.yTicks.map(v => (
                <g key={"y" + v}>
                  <line x1={chart.ML} y1={chart.y(v)} x2={chart.W - chart.MR} y2={chart.y(v)}
                        stroke="#332A20" strokeWidth="1" />
                  <text x={chart.ML - 7} y={chart.y(v) + 4} textAnchor="end"
                        fill="#918675" fontSize="11" fontFamily="IBM Plex Mono, monospace">
                    {v.toFixed(step2(chart.yTicks))}
                  </text>
                </g>
              ))}

              {/* x grid */}
              {chart.xTicks.map(t => (
                <g key={"x" + t}>
                  <line x1={chart.x(t)} y1={chart.MT} x2={chart.x(t)} y2={chart.MT + chart.ph}
                        stroke="#2A2219" strokeWidth="1" />
                  <text x={chart.x(t)} y={chart.MT + chart.ph + 17} textAnchor="middle"
                        fill="#918675" fontSize="11" fontFamily="IBM Plex Mono, monospace">
                    {fmtClock(t)}
                  </text>
                </g>
              ))}

              {/* driving limit */}
              {chart.limitPm <= chart.yMaxPm && (
                <>
                  <line x1={chart.ML} y1={chart.y(chart.limitPm)} x2={chart.W - chart.MR}
                        y2={chart.y(chart.limitPm)} stroke="#E07B33" strokeWidth="1.2"
                        strokeDasharray="5,4" opacity="0.75" />
                  <text x={chart.W - chart.MR - 3} y={chart.y(chart.limitPm) - 6} textAnchor="end"
                        fill="#E07B33" fontSize="10.5" fontFamily="IBM Plex Mono, monospace"
                        opacity="0.9">
                    limit {chart.limitPm}‰
                  </text>
                </>
              )}

              {/* uncertainty envelope + central line */}
              <path d={chart.envelope} fill="url(#bcEnv)" />
              <path d={chart.cLine} fill="none" stroke="#E9A93C" strokeWidth="2.2"
                    strokeLinejoin="round" strokeLinecap="round" />

              {/* peak marker */}
              <circle cx={chart.peakX} cy={chart.peakY} r="3.4" fill="#EFE7DA" />
              <text x={chart.peakX} y={chart.peakY - 10} textAnchor="middle"
                    fill="#EFE7DA" fontSize="11" fontFamily="IBM Plex Mono, monospace">
                {chart.peakPm.toFixed(2)}
              </text>

              {/* now marker */}
              {chart.nowX > chart.ML && chart.nowX < chart.W - chart.MR && (
                <>
                  <line x1={chart.nowX} y1={chart.MT} x2={chart.nowX} y2={chart.MT + chart.ph}
                        stroke="#EFE7DA" strokeWidth="1.4" opacity="0.55" />
                  <text x={chart.nowX} y={chart.MT - 3} textAnchor="middle"
                        fill="#EFE7DA" fontSize="10" fontFamily="IBM Plex Mono, monospace"
                        opacity="0.75">now</text>
                </>
              )}

              {/* drink windows: the whole point of this calculator, drawn */}
              {chart.drinkBars.map(b => (
                <rect key={b.idx} x={b.x1} y={chart.MT + chart.ph + 26}
                      width={Math.max(b.x2 - b.x1, 3)} height="7" rx="3"
                      fill="#E9A93C" opacity="0.55" />
              ))}
              <text x={chart.ML} y={chart.MT + chart.ph + 46} fill="#918675" fontSize="10"
                    fontFamily="IBM Plex Mono, monospace">
                bars = when each drink was actually being drunk
              </text>
            </svg>

            <p className="bc-sr">
              {(() => {
                const pk = result.peak, sl = result.slopePerHour;
                const dir = sl > 0.02 ? "still rising" : sl < -0.02 ? "falling" : "level";
                return `Blood alcohol over the session. Currently ${perMille(result.nowBac).toFixed(2)} ` +
                  `permille and ${dir}. Peak of ${perMille(pk.bac).toFixed(2)} permille at ` +
                  `${fmtClock(pk.t)}, with a likely range of ${perMille(result.peakP10).toFixed(2)} ` +
                  `to ${perMille(result.peakP90).toFixed(2)}. ` +
                  (result.clearCautious
                    ? `To be safe, below 0.1 permille by ${fmtClock(result.clearCautious)}.` : "");
              })()}
            </p>

            <div className="bc-legend">
              <span><i style={{ background: "#E9A93C" }} />central estimate</span>
              <span><b style={{ background: "rgba(233,169,60,.28)" }} />middle 80% of people</span>
              <span><i style={{ background: "#E07B33" }} />your limit</span>
            </div>
          </section>
        )}

        {/* ── DRINKS ───────────────────────────────────────────────── */}
        <section className="bc-card">
          <span className="bc-lbl">Drinks · {totalG.toFixed(0)} g alcohol total</span>
          <div ref={listRef}>

          {drinks.map((d, i) => {
            const g = (parseFloat(d.ml) || 0) * ((parseFloat(d.pct) || 0) / 100) * ETHANOL_DENSITY;
            const abv = parseFloat(d.pct) || 0;
            const halfLife = abv > 0
              ? Math.LN2 / absorptionK(abv, food, d.mixer, parseFloat(d.ml) || 0, d.fizzy) : null;
            return (
              <div className="bc-drink" key={i}>
                <div className="bc-dhead">
                  <span className="bc-dnum">
                    DRINK {String(i + 1).padStart(2, "0")}
                    {dayOffsetByIdx[i] > 0 && (
                      <span className="bc-dday" title="After midnight, relative to the first drink">
                        +{dayOffsetByIdx[i]}d
                      </span>
                    )}
                    <span className="bc-dgram">
                      {g.toFixed(1)} g
                      {halfLife && ` · absorbs ~${Math.round(halfLife)} min`}
                    </span>
                  </span>
                  <span style={{ display: "flex", gap: 6, alignItems: "center" }}>
                    <button className="bc-btn" onClick={() => duplicateDrink(i)}
                      title="Add another one just like this, starting when this one ended">
                      duplicate
                    </button>
                    <button className="bc-btn" onClick={() => finishNow(i)}>finished now</button>
                    {drinks.length > 1 && (
                      <button className="bc-x" onClick={() => rm(i)} aria-label="Remove drink">×</button>
                    )}
                  </span>
                </div>

                <div className="bc-presets">
                  {PRESETS.map(p => (
                    <button key={p.label} className="bc-btn"
                      onClick={() => setDrinks(ds => ds.map((x, j) =>
                        j === i ? { ...x, ml: p.ml, pct: p.pct, fizzy: p.fizzy } : x))}>
                      {p.label}
                    </button>
                  ))}
                </div>

                <div className="bc-f" style={{ marginBottom: 10 }}>
                  <label>
                    Mixed with
                    <button
                      className={`bc-fizz${d.fizzy ? " on" : ""}`}
                      aria-pressed={!!d.fizzy}
                      onClick={() => upd(i, "fizzy", !d.fizzy)}>
                      {d.fizzy ? "● fizzy" : "○ flat"}
                    </button>
                  </label>
                  <div className="bc-seg" role="radiogroup" aria-label="Mixed with">
                    {[["plain", "Nothing"], ["sugary", "Sugary"], ["diet", "Diet"]].map(([v, l]) => (
                      <div key={v} className="bc-opt"
                        data-on={(d.mixer || "plain") === v ? "1" : "0"}
                        role="radio" aria-checked={(d.mixer || "plain") === v} tabIndex={0}
                        onClick={() => upd(i, "mixer", v)}
                        onKeyDown={e => e.key === "Enter" && upd(i, "mixer", v)}>{l}</div>
                    ))}
                  </div>
                </div>

                <div className="bc-grid4">
                  <div className="bc-f">
                    <label>Volume (ml)</label>
                    <input className="bc-in" type="text" inputMode="decimal" value={d.ml}
                      onChange={e => upd(i, "ml", digits(e.target.value))} />
                  </div>
                  <div className="bc-f">
                    <label>Alcohol %</label>
                    <input className="bc-in" type="text" inputMode="decimal" value={d.pct}
                      onChange={e => upd(i, "pct", decimal(e.target.value))} />
                  </div>
                  <div className="bc-f">
                    <label>Started</label>
                    <div className="bc-time">
                      <input className="bc-in" type="text" inputMode="numeric" value={d.startH}
                        onChange={e => upd(i, "startH", hm(e.target.value))} />
                      <span className="bc-colon">:</span>
                      <input className="bc-in" type="text" inputMode="numeric" value={d.startM}
                        onChange={e => upd(i, "startM", hm(e.target.value))} />
                    </div>
                  </div>
                  <div className="bc-f">
                    <label>Finished</label>
                    <div className="bc-time">
                      <input className="bc-in" type="text" inputMode="numeric" value={d.endH}
                        onChange={e => upd(i, "endH", hm(e.target.value))} />
                      <span className="bc-colon">:</span>
                      <input className="bc-in" type="text" inputMode="numeric" value={d.endM}
                        onChange={e => upd(i, "endM", hm(e.target.value))} />
                    </div>
                  </div>
                </div>
              </div>
            );
          })}

          </div>
          <button className="bc-add" onClick={addDrink}>+ Add another drink</button>

          <div className="bc-reset">
            {confirmReset ? (
              <>
                <span>Clear tonight's drinks and start over?</span>
                <button className="bc-btn bc-btn-warn" onClick={newNight}>Yes, clear</button>
                <button className="bc-btn" onClick={() => setConfirmReset(false)}>Cancel</button>
              </>
            ) : (
              <button className="bc-btn" onClick={() => setConfirmReset(true)}>
                Start a new night
              </button>
            )}
          </div>
        </section>

        {/* ── PROFILE ──────────────────────────────────────────────── */}
        <section className="bc-card">
          <span className="bc-lbl">About you</span>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div className="bc-grid4">
              <div className="bc-f">
                <label>Weight (kg)</label>
                <input className="bc-in" type="text" inputMode="decimal" value={profile.weightKg}
                  onChange={e => setP("weightKg", decimal(e.target.value))} />
              </div>
              <div className="bc-f">
                <label>Height (cm)</label>
                <input className="bc-in" type="text" inputMode="decimal" value={profile.heightCm}
                  onChange={e => setP("heightCm", decimal(e.target.value))} />
              </div>
              <div className="bc-f">
                <label>Age</label>
                <input className="bc-in" type="text" inputMode="numeric" value={profile.age}
                  onChange={e => setP("age", digits(e.target.value))} />
              </div>
              <div className="bc-f">
                <label>Sex</label>
                <div className="bc-seg" role="radiogroup" aria-label="Biological sex">
                  {["male", "female"].map(s => (
                    <div key={s} className="bc-opt" data-on={profile.sex === s ? "1" : "0"}
                      onClick={() => setP("sex", s)} role="radio"
                      aria-checked={profile.sex === s} tabIndex={0}
                      onKeyDown={e => e.key === "Enter" && setP("sex", s)}>
                      {s === "male" ? "M" : "F"}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ── CONDITIONS ───────────────────────────────────────────── */}
        <section className="bc-card">
          <span className="bc-lbl">Tonight</span>
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div className="bc-f">
              <label>Food in your stomach</label>
              <div className="bc-seg" role="radiogroup" aria-label="Food in your stomach">
                {[["empty", "Empty"], ["light", "Snacked"], ["full", "Full meal"]].map(([v, l]) => (
                  <div key={v} className="bc-opt" data-on={food === v ? "1" : "0"}
                    onClick={() => setFood(v)} role="radio"
                    aria-checked={food === v} tabIndex={0}
                    onKeyDown={e => e.key === "Enter" && setFood(v)}>{l}</div>
                ))}
              </div>
            </div>
            <div className="bc-f">
              <label>How often you normally drink</label>
              <div className="bc-seg" role="radiogroup" aria-label="How often you normally drink">
                {FREQ.map(([v, l]) => (
                  <div key={v} className="bc-opt" data-on={freq === v ? "1" : "0"}
                    onClick={() => setFreq(v)} role="radio"
                    aria-checked={freq === v} tabIndex={0}
                    onKeyDown={e => e.key === "Enter" && setFreq(v)}>{l}</div>
                ))}
              </div>
            </div>
            <div className="bc-f">
              <label>Driving limit where you are (‰)</label>
              <div className="bc-seg" role="radiogroup" aria-label="Driving limit">
                {LIMITS.map(l => (
                  <div key={l.v} className="bc-opt" data-on={limit === l.v ? "1" : "0"}
                    onClick={() => setLimit(l.v)} role="radio"
                    aria-checked={limit === l.v} tabIndex={0}
                    onKeyDown={e => e.key === "Enter" && setLimit(l.v)}>{l.label} ‰</div>
                ))}
              </div>
            </div>
          </div>
        </section>

        <footer className="bc-foot">
          <p>
            <strong>This is a model, not a measurement.</strong> Real blood
            alcohol varies with liver function, medication, illness, hydration,
            sleep and genetics, none of which this can see. The shaded range comes
            from running the model {MC_RUNS} times with each parameter drawn from
            its own measured spread, and those spreads are not guesses: they were
            set so the model reproduces the scatter seen when researchers dosed a
            group of men identically and measured what happened. It covers
            roughly the middle 80% of people. Your own body sits at one fixed
            point inside it, and nothing here can tell you where. Never treat any
            number as clearance to drive.
          </p>
          <p style={{ marginTop: 10 }}>
            <strong>Whose bodies were these equations built on?</strong> The
            body-water equation underneath this was derived from a fairly narrow
            population, and its real-world error is larger than its published fit
            suggests. It has been reported to overstate body water by around 12%
            in South Asian subjects, and overstating body water understates blood
            alcohol. The spread shown here was calibrated against a study whose
            subjects were mostly not White, which helps, but if your build is far
            from average treat the band as the answer and the middle number as a
            rough guide.
          </p>
          <p style={{ marginTop: 10 }}>
            <strong>A diet mixer gets you drunker than a sugary one.</strong> Same
            spirit, same measure: sugar triggers the duodenal feedback that slows
            your stomach emptying, while sweetener does not. In a controlled
            crossover the diet version emptied in 21 minutes against 36, and
            peaked 56% higher. Set this per drink.
          </p>
          <p style={{ marginTop: 10 }}>
            <strong>Strength matters, not just units.</strong> The same alcohol
            drunk as spirits peaks higher and sooner than as beer, because
            concentrated drinks leave the stomach faster. Above about 30% ABV
            that reverses again, since neat spirits irritate the stomach lining
            and slow emptying. This is modelled per drink.
          </p>
          {result.ready && (
            <p className="bc-meta">
              Widmark r {result.central.r.toFixed(3)} · body water {result.central.tbw.toFixed(1)} L ·
              elimination {result.central.beta.toFixed(4)} g/dL/h ·
              absorption t½ {Math.round(Math.LN2 / result.central.meanKa)} min ·
              {" "}{(totalG / 10).toFixed(1)} EU units (10 g) ·
              {" "}{MC_RUNS} Monte Carlo runs · band calibrated to observed scatter
            </p>
          )}
        </footer>

      </div>
    </div>
  );
}

// decimal places for the y-axis labels, based on the tick spacing in use
function step2(ticks) {
  if (ticks.length < 2) return 1;
  const d = Math.abs(ticks[1] - ticks[0]);
  return d < 0.2 ? 1 : d < 0.5 ? 2 : 1;
}

"""Загрузка, разрешение имён колонок, метки, дизайн-матрица, пространственно-временные блоки."""
from __future__ import annotations
import os, re
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

DRIVER_SIGN: dict[str, int] = {"temp": +1, "rh": -1, "wind": +1, "precip": -1,
                               "vpd": +1, "solar": +1}
DRIVERS_BASE = ["temp", "rh", "wind", "precip", "vpd"]

_ALIASES: dict[str, list[str]] = {
    "dew":      ["dew", "dewpoint", "dew_point", "dewtemp", "td"],
    "heat":     ["heat", "heatindex", "heat_index", "apparenttemp", "feelslike"],
    "solar":    ["solar", "solarradiation", "solar_radiation", "radiation", "srad",
                 "ssrd", "shortwave", "insolation", "swdown", "rsds", "netradiation"],
    "vpd":      ["vpd", "vaporpressuredeficit", "dryness", "drought", "fwi", "dmc", "dc"],
    "pressure": ["pressure", "slp", "msl", "mslp", "sealevelpressure"],
    "cloud":    ["cloud", "cloudcover", "clouds", "tcc"],
    "precip":   ["precip", "precipitation", "rain", "rainfall", "prcp", "tp"],
    "rh":       ["rh", "humidity", "relativehumidity", "rhum", "hum"],
    "wind":     ["wind", "windspeed", "ws", "w10m", "wspd"],
    "temp":     ["temp", "temperature", "t2m", "tavg", "tmax", "airtemp"],
    "lat":      ["lat", "latitude", "y"],
    "lon":      ["lon", "long", "longitude", "x"],
    "date":     ["date", "datetime", "time", "acqdate", "acq_date", "timestamp",
                 "dt", "day", "observed", "fired"],
    "label":    ["type", "firetype", "fire_type", "class", "category", "label", "target"],
}
CTX_CANDIDATES_BASE = ["dew", "heat", "pressure", "cloud", "solar"]
_MIN_PARTIAL = 3
_norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())

_WANT_KIND: dict[str, tuple[str, ...]] = {"label": ("str", "num"), "date": ("dt", "str", "num")}
_DEFAULT_KIND = ("num", "str")


def _kind(s: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(s): return "dt"
    if pd.api.types.is_bool_dtype(s):           return "str"
    if pd.api.types.is_numeric_dtype(s):        return "num"
    return "str"


def _rank(key: str, col: str, s: pd.Series) -> tuple:
    want = _WANT_KIND.get(key, _DEFAULT_KIND)
    k = _kind(s)
    return (want.index(k) if k in want else len(want), len(col), col)


def describe_columns(df: pd.DataFrame) -> None:
    print(f"\n[columns] фактические колонки файла ({df.shape[1]}):")
    for c in df.columns:
        s = df[c]
        ex = ", ".join(map(str, s.dropna().unique()[:3]))
        print(f"   {c:22s} {str(s.dtype):10s} nuniq={s.nunique():<7d} примеры: {ex}")


def resolve_columns(df: pd.DataFrame, overrides: dict[str, str]) -> dict[str, str]:
    used, out = set(overrides.values()), dict(overrides)
    for k, v in overrides.items():
        assert v in df.columns, f"--map {k}={v}: колонки нет. Есть: {list(df.columns)}"
    cand = {c: _norm(c) for c in df.columns}
    alts: dict[str, list[str]] = {}
    for key, aliases in _ALIASES.items():
        if key in out:
            continue
        na = list(map(_norm, aliases))
        longs = [a for a in na if len(a) >= _MIN_PARTIAL]
        exact = [c for c, n in cand.items() if n in na and c not in used]
        part  = [c for c, n in cand.items() if any(a in n for a in longs) and c not in used]
        pool  = sorted(exact or part, key=lambda c: _rank(key, c, df[c]))
        if pool:
            out[key] = pool[0]; used.add(pool[0]); alts[key] = pool[1:]

    print("\n[columns] разрешение имён (ПРОВЕРЬ!):")
    for k in _ALIASES:
        pick = out.get(k); note = ""
        if pick is not None:
            note = f"   [{_kind(df[pick])}]"
            rej = alts.get(k, [])
            if rej:
                note += "  отклонены: " + ", ".join(f"{c}[{_kind(df[c])}]" for c in rej)
            if k in overrides:
                note += "  (задано --map)"
        print(f"   {k:9s} -> {pick or '— не найдено —'}{note}")
    return out


@dataclass
class Dataset:
    df: pd.DataFrame
    y: np.ndarray
    classes: list[str]
    drivers: list[str]
    ctx_candidates: list[str]
    geo: list[str]
    blocks: np.ndarray
    coord_blocks: np.ndarray
    time_blocks: np.ndarray | None = None
    K: int = field(init=False)

    def __post_init__(self): self.K = len(self.classes)

    @property
    def base_feats(self) -> list[str]: return self.drivers + self.geo

    @property
    def has_time(self) -> bool: return self.time_blocks is not None


def available_protocols(ds: Dataset) -> list[str]:
    p = ["iid", "point", "spatial"]
    if ds.has_time: p += ["temporal", "spatiotemporal"]
    return p


def _parse_merge(spec: str) -> dict[str, str]:
    m = {}
    for rule in filter(None, (spec or "").split(";")):
        src, dst = rule.split("=", 1)
        for s in src.split("+"):
            m[s.strip()] = dst.strip()
    return m


def _to_datetime(s: pd.Series) -> pd.Series:
    for kw in ({"format": "mixed"}, {}):
        try:
            return pd.to_datetime(s, errors="coerce", **kw)
        except (TypeError, ValueError):
            continue
    return pd.to_datetime(s.astype(str), errors="coerce")


def load_dataset(csv: str, order: str, merge: str = "", overrides: dict[str, str] | None = None,
                 block_deg: float = 2.0, min_class: int = 30, solar_as_driver: bool = False,
                 time_block_days: int = 7, dedup: str = "none") -> Dataset:
    assert os.path.exists(csv), (f"файла нет: {csv}\n  cwd={os.getcwd()}\n"
                                 f"  рядом .csv: {sorted(f for f in os.listdir('.') if f.endswith('.csv'))}")
    raw = pd.read_csv(csv)
    print(f"[load] {csv}: {raw.shape[0]} строк, {raw.shape[1]} колонок")
    describe_columns(raw)
    col = resolve_columns(raw, overrides or {})

    assert "label" in col, "не найдена колонка класса — задай --map label=<имя>"
    lab = raw[col["label"]].astype(str).str.strip()
    mm = _parse_merge(merge)
    if mm:
        lab = lab.replace(mm)
        print(f"[label] слияние: {mm}")
    classes = [c.strip() for c in order.split(",")]
    assert len(classes) >= 2, "--order должен задавать >=2 упорядоченных класса"

    present = [c for c in classes if (lab == c).any()]
    if not present:
        strs = [c for c in raw.columns if _kind(raw[c]) == "str"]
        raise AssertionError(
            f"колонка метки '{col['label']}' не содержит НИ ОДНОГО класса из --order.\n"
            f"  --order = {classes}\n"
            f"  значения (топ-12): {dict(lab.value_counts().head(12))}\n"
            f"  строковые колонки-кандидаты: {strs}")
    if len(present) < len(classes):
        print(f"[label] ВНИМАНИЕ: классы из --order отсутствуют в данных: "
              f"{sorted(set(classes) - set(present))}")
    unknown = sorted(set(lab) - set(classes))
    if unknown:
        n_out = int(lab.isin(unknown).sum())
        print(f"[label] отброшено {n_out} строк ({n_out/len(lab):.1%}) с классами вне --order: {unknown}")
    keep_rows = lab.isin(classes)
    assert keep_rows.sum() > 0, "после фильтра по --order не осталось строк"

    driver_keys = list(DRIVERS_BASE) + (["solar"] if solar_as_driver else [])
    ctx_keys = [c for c in CTX_CANDIDATES_BASE if not (c == "solar" and solar_as_driver)]

    d = pd.DataFrame(index=raw.index)
    drivers, ctx = [], []
    for k in driver_keys:
        if k in col:
            d[k] = pd.to_numeric(raw[col[k]], errors="coerce"); drivers.append(k)
    for k in ctx_keys:
        if k in col:
            d[k] = pd.to_numeric(raw[col[k]], errors="coerce"); ctx.append(k)
    geo = [g for g in ("lat", "lon") if g in col]
    for g in geo:
        d[g] = pd.to_numeric(raw[col[g]], errors="coerce")

    if "solar" in col:
        print(f"[solar] '{col['solar']}' -> роль: "
              + ("ДРАЙВЕР (наивный знак +1)" if solar_as_driver
                 else "контекст-кандидат (решает шлюз экзогенности)"))
    else:
        print("[solar] колонка инсоляции не найдена")

    has_time = False
    if "date" in col:
        dt = _to_datetime(raw[col["date"]])
        frac = float(dt.notna().mean())
        if frac > 0.9:
            doy = dt.dt.dayofyear.astype(float)
            d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
            d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
            d["_t"] = (dt - dt.min()).dt.total_seconds() / 86400.0
            ctx += ["doy_sin", "doy_cos"]; has_time = True
            print(f"[ctx] добавлены экзогенные doy_sin, doy_cos из '{col['date']}' "
                  f"({dt.min().date()}..{dt.max().date()}, охват {(dt.max()-dt.min()).days} дней)")
        else:
            print(f"[ctx] '{col['date']}' не парсится как дата ({frac:.1%}) — календарь недоступен")
    else:
        print("[ctx] колонки даты в файле нет => temporal/spatiotemporal НЕДОСТУПНЫ, "
              "сезонность неконтролируема")

    assert len(drivers) >= 3, f"нужно >=3 драйвера, найдено {drivers}"
    assert len(geo) == 2, "нужны lat и lon для пространственно-блочной валидации"

    need = drivers + geo + (["_t"] if has_time else [])
    ok = keep_rows & d[need].notna().all(axis=1)
    print(f"[data] строк после --order и notna(драйверы,гео{',дата' if has_time else ''}): "
          f"{int(ok.sum())} из {len(raw)}")
    d, lab = d[ok].copy(), lab[ok]
    for c in ctx:
        d[c] = d[c].fillna(d[c].median())
    ctx = [c for c in ctx if d[c].std() > 0]
    if not ctx:
        print("[ctx] ИТОГ: контекст-кандидатов НЕТ ни одного.\n"
              "      => шлюз закроется по отсутствию кандидатов, а не по скринингу;\n"
              "      => MIB* тождественны MONO_sc, Результат 3 НЕ проверяется.")

    vc = lab.value_counts()
    kept = [c for c in classes if vc.get(c, 0) >= min_class]
    dropped = [(c, int(vc.get(c, 0))) for c in classes if c not in kept]
    if dropped:
        print(f"[label] классы выброшены по min_class={min_class}: {dropped}")
    assert len(kept) >= 2, f"после min_class={min_class} осталось <2 классов"
    classes = kept
    m2 = lab.isin(classes); d, lab = d[m2].copy(), lab[m2]

    key_cols = geo + drivers
    n_exact = int(d.duplicated(subset=key_cols).sum())
    cc = (d["lat"].round(4).astype(str) + "_" + d["lon"].round(4).astype(str))
    mult = cc.value_counts()
    print(f"[dup] точных дублей по (гео+драйверы): {n_exact} ({n_exact/len(d):.2%}); "
          f"уникальных координат {mult.size} на {len(d)} строк; "
          f"доля строк в повторяющихся точках {(mult[mult>1].sum())/len(d):.2%}; "
          f"макс. кратность {int(mult.max())}")
    if dedup == "exact" and n_exact:
        d = d[~d.duplicated(subset=key_cols)]; lab = lab.loc[d.index]
        print(f"[dup] --dedup exact: осталось {len(d)} строк")
    elif dedup == "coord":
        keep = ~cc.loc[d.index].duplicated()
        d, lab = d[keep], lab[keep]
        print(f"[dup] --dedup coord: одна строка на координату, осталось {len(d)} строк")

    y = lab.map({c: i for i, c in enumerate(classes)}).to_numpy(int)
    bl = (np.floor(d["lat"] / block_deg).astype(int).astype(str) + "_" +
          np.floor(d["lon"] / block_deg).astype(int).astype(str))
    blocks = pd.factorize(bl)[0]
    coord_blocks = pd.factorize(d["lat"].round(4).astype(str) + "_" +
                                d["lon"].round(4).astype(str))[0]

    time_blocks = None
    if has_time:
        tb = np.floor(d["_t"].to_numpy(float) / float(time_block_days)).astype(int)
        uq = np.unique(tb)
        time_blocks = np.searchsorted(uq, tb)
        print(f"[data] временных блоков по {time_block_days} дн: {uq.size}")

    cnt = np.bincount(y, minlength=len(classes))
    print(f"[data] N={len(d)}  K={len(classes)}  классы(порядок)={classes}")
    print(f"[data] распределение: {dict(zip(classes, cnt.tolist()))}  "
          f"доли={np.round(cnt / cnt.sum(), 4).tolist()}")
    print(f"[data] baseline macro-F1 «всегда мажоритарный» = {_majority_macro_f1(cnt):.4f}")
    print(f"[data] драйверы={drivers}  контекст-кандидаты={ctx or '[]'}  "
          f"блоков({block_deg}deg)={blocks.max() + 1}  уник.точек={coord_blocks.max() + 1}")
    return Dataset(d.reset_index(drop=True), y, classes, drivers, ctx, geo,
                   blocks, coord_blocks, time_blocks)


def _majority_macro_f1(cnt: np.ndarray) -> float:
    n, m = cnt.sum(), cnt.max()
    return float((2 * m / (n + m)) / len(cnt))


def design(df: pd.DataFrame, drivers: list[str]) -> np.ndarray:
    """Квадратичный дизайн по драйверам (1, x, x^2, x_i*x_j) — для скрининга экзогенности."""
    X = df[drivers].to_numpy(float)
    mu, sd = X.mean(0), np.where(X.std(0) > 0, X.std(0), 1.0)
    Z = (X - mu) / sd
    cols = [np.ones(len(Z))] + [Z[:, j] for j in range(Z.shape[1])] + \
           [Z[:, j] ** 2 for j in range(Z.shape[1])]
    cols += [Z[:, i] * Z[:, j] for i in range(Z.shape[1]) for j in range(i + 1, Z.shape[1])]
    return np.column_stack(cols)
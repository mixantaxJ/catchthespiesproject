import re
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(".")

AIRPORT_COORDS = {
    "SVO": (55.9726, 37.4146, "Moscow SVO"),
    "DME": (55.4088, 37.9063, "Moscow DME"),
    "VKO": (55.5915, 37.2615, "Moscow VKO"),
    "LED": (59.8003, 30.2625, "St. Petersburg"),
    "VVO": (43.3990, 132.1480, "Vladivostok"),
    "OVB": (55.0126, 82.6507, "Novosibirsk"),
    "SVX": (56.7431, 60.8027, "Yekaterinburg"),
    "KJA": (56.1729, 92.4933, "Krasnoyarsk"),
    "MMK": (68.7817, 32.7508, "Murmansk"),
    "VOG": (48.7825, 44.3455, "Volgograd"),
    "KZN": (55.6062, 49.2787, "Kazan"),
    "AER": (43.4499, 39.9566, "Sochi"),
    "ROV": (47.2582, 39.8181, "Rostov"),
    "UFA": (54.5575, 55.8744, "Ufa"),
    "KRR": (45.0347, 39.1705, "Krasnodar"),
    "IKT": (52.2680, 104.3890, "Irkutsk"),
    "KHV": (48.5280, 135.1880, "Khabarovsk"),
    "PKC": (53.1678, 158.4539, "Petropavlovsk"),
    "UUS": (46.8886, 142.7178, "Yuzhno-Sakhalinsk"),
    "OMS": (54.9670, 73.3100, "Omsk"),
    "CEK": (55.3058, 61.5033, "Chelyabinsk"),
    "SAM": (53.5049, 50.1643, "Samara"),
    "KUF": (53.5049, 50.1643, "Samara"),
    "GOJ": (56.2301, 43.7840, "Nizhny Novgorod"),
    "MRV": (44.2251, 43.0819, "Mineralnye Vody"),
    "TJM": (57.1896, 65.3243, "Tyumen"),
    "PEE": (57.9145, 56.0212, "Perm"),
    "AAQ": (45.0021, 37.3473, "Anapa"),
    "SIP": (45.0522, 33.9751, "Simferopol"),
    "KGD": (54.8900, 20.5926, "Kaliningrad"),
    "ARH": (64.6003, 40.7167, "Arkhangelsk"),
    "SCW": (61.6470, 50.8450, "Syktyvkar"),
    "CDG": (49.0097, 2.5479, "Paris CDG"),
    "AMS": (52.3105, 4.7683, "Amsterdam"),
    "FCO": (41.8003, 12.2389, "Rome FCO"),
    "MXP": (45.6306, 8.7281, "Milan MXP"),
    "PRG": (50.1008, 14.2600, "Prague"),
    "OTP": (44.5711, 26.0850, "Bucharest"),
    "LHR": (51.4700, -0.4543, "London LHR"),
    "MAD": (40.4983, -3.5676, "Madrid"),
    "FRA": (50.0379, 8.5622, "Frankfurt"),
    "MUC": (48.3537, 11.7750, "Munich"),
    "ICN": (37.4602, 126.4407, "Seoul ICN"),
    "PEK": (40.0799, 116.6031, "Beijing"),
    "PVG": (31.1443, 121.8083, "Shanghai PVG"),
    "CAN": (23.3924, 113.2988, "Guangzhou"),
    "SZX": (22.6393, 113.8107, "Shenzhen"),
    "WUH": (30.7838, 114.2081, "Wuhan"),
    "XMN": (24.5440, 118.1277, "Xiamen"),
    "HND": (35.5494, 139.7798, "Tokyo HND"),
    "NRT": (35.7720, 140.3929, "Tokyo NRT"),
    "DEL": (28.5562, 77.1000, "New Delhi"),
    "DXB": (25.2532, 55.3657, "Dubai"),
    "JED": (21.6796, 39.1565, "Jeddah"),
    "RUH": (24.9576, 46.6988, "Riyadh"),
    "TLV": (32.0055, 34.8854, "Tel Aviv"),
    "IST": (40.9769, 28.8146, "Istanbul"),
    "BKK": (13.6900, 100.7501, "Bangkok"),
    "SGN": (10.8188, 106.6520, "Ho Chi Minh"),
    "HAN": (21.2212, 105.8072, "Hanoi"),
    "JFK": (40.6413, -73.7781, "New York JFK"),
    "LGA": (40.7769, -73.8740, "New York LGA"),
    "EWR": (40.6895, -74.1745, "Newark"),
    "ATL": (33.6407, -84.4277, "Atlanta"),
    "DTW": (42.2162, -83.3554, "Detroit"),
    "MSP": (44.8848, -93.2223, "Minneapolis"),
    "SLC": (40.7899, -111.9791, "Salt Lake City"),
    "LAX": (33.9416, -118.4085, "Los Angeles"),
    "SFO": (37.6213, -122.3790, "San Francisco"),
    "SEA": (47.4502, -122.3088, "Seattle"),
    "BOS": (42.3656, -71.0096, "Boston"),
    "ORD": (41.9742, -87.9073, "Chicago"),
    "MIA": (25.7959, -80.2870, "Miami"),
    "IAD": (38.9531, -77.4565, "Washington"),
    "MEX": (19.4363, -99.0721, "Mexico City"),
    "EZE": (-34.8222, -58.5358, "Buenos Aires"),
}

TRANSLIT_DICT = {
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E", "Ё": "E",
    "Ж": "ZH", "З": "Z", "И": "I", "Й": "I", "К": "K", "Л": "L", "М": "M",
    "Н": "N", "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T", "У": "U",
    "Ф": "F", "Х": "KH", "Ц": "TS", "Ч": "CH", "Ш": "SH", "Щ": "SHCH",
    "Ъ": "", "Ы": "Y", "Ь": "", "Э": "E", "Ю": "IU", "Я": "IA",
}


def transliterate_and_normalize(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    s = "".join(TRANSLIT_DICT.get(ch, ch) for ch in text.strip().upper())
    tokens = []
    for tok in re.split(r"[\s_]+", s):
        t = (
            tok.replace("X", "KS")
            .replace("YO", "IO")
            .replace("YA", "IA")
            .replace("YU", "IU")
            .replace("YE", "E")
            .replace("IY", "II")
            .replace("YY", "II")
            .replace("Y", "I")
            .replace("KH", "H")
            .replace("TS", "C")
            .replace("TC", "C")
        )
        t = re.sub(r"[^A-Z]", "", t)
        if len(t) > 1:
            tokens.append(t)
    if len(tokens) >= 2:
        return "_".join(sorted(tokens[:2]))
    return tokens[0] if tokens else ""


@st.cache_data(show_spinner=False)
def load_summary() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_DIR / "yoyoflot_passengers_summary.csv",
        dtype={"primary_doc": str, "primary_ff_card": str},
    )
    if "known_baggage_flights" not in df.columns:
        df["known_baggage_flights"] = df["total_flights"]
    if "baggage_known_pct" not in df.columns:
        df["baggage_known_pct"] = (
            df["known_baggage_flights"] / df["total_flights"].clip(lower=1) * 100
        ).round(1)
    if "no_baggage_pct" not in df.columns:
        denom = df["known_baggage_flights"].replace(0, np.nan)
        df["no_baggage_pct"] = (df["no_baggage_flights"] / denom * 100).round(1)
    df["no_baggage_pct"] = df["no_baggage_pct"].fillna(0.0)
    df["baggage_known_pct"] = df["baggage_known_pct"].fillna(0.0)

    if "has_doc_conflict" not in df.columns:
        df["has_doc_conflict"] = False
    if "has_ff_conflict" not in df.columns:
        df["has_ff_conflict"] = False
    if "identity_score" not in df.columns:
        df["identity_score"] = (
            df["has_doc_conflict"].astype(int) * 100
            + df["has_ff_conflict"].astype(int) * 80
        )
    if "no_baggage_signal" not in df.columns:
        df["no_baggage_signal"] = (
            df["no_baggage_pct"] * df["baggage_known_pct"] / 100.0
        ).round(1)
    if "courier_score" not in df.columns:
        df["courier_score"] = (
            df["total_flights"] * 1.2
            + df["quick_turnarounds"] * 4.5
            + df["no_baggage_signal"] * 0.2
        ).round(1)
    if "risk_score" not in df.columns:
        df["risk_score"] = (df["identity_score"] + df["courier_score"]).round(1)
    return df


@st.cache_data(show_spinner=False)
def load_flights() -> pd.DataFrame:
    candidates = [
        DATA_DIR / "yoyoflot_flights_analytics.csv",
        DATA_DIR / "yoyoflot_flights_analytics.csv.gz",
        DATA_DIR / "yoyoflot_flights_merged_clean.csv",
        DATA_DIR / "yoyoflot_flights_merged_clean.csv.gz",
    ]
    target = next((p for p in candidates if p.exists()), candidates[-1])
    return pd.read_csv(
        target,
        dtype={
            "doc_clean": str,
            "ticket_number": str,
            "ff_card": str,
            "booking_code": str,
        },
        low_memory=False,
    )


@st.cache_data(show_spinner=False)
def get_conflict_tables(df_fl: pd.DataFrame):
    doc_stats = (
        df_fl.dropna(subset=["doc_clean"])
        .groupby("doc_clean")
        .agg(
            n_names=("full_name_norm", "nunique"),
            n_birth_dates=("birth_date", "nunique"),
            n_profiles=("person_id_final", "nunique"),
        )
    )
    doc_conflicts = doc_stats[
        (doc_stats["n_names"] > 1)
        | (doc_stats["n_birth_dates"] > 1)
        | (doc_stats["n_profiles"] > 1)
    ].sort_values(["n_profiles", "n_names", "n_birth_dates"], ascending=False)

    ff_stats = (
        df_fl.dropna(subset=["ff_card"])
        .groupby("ff_card")
        .agg(
            n_names=("full_name_norm", "nunique"),
            n_birth_dates=("birth_date", "nunique"),
            n_profiles=("person_id_final", "nunique"),
        )
    )
    ff_conflicts = ff_stats[
        (ff_stats["n_names"] > 1)
        | (ff_stats["n_birth_dates"] > 1)
        | (ff_stats["n_profiles"] > 1)
    ].sort_values(["n_profiles", "n_names", "n_birth_dates"], ascending=False)

    return doc_conflicts, ff_conflicts


def render_etl_tab():
    st.subheader("Воронка интеграции данных")
    col_sankey, col_sources = st.columns([3, 2])

    with col_sankey:
        fig_sankey = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=[
                            "SkyTeam YAML (1.69M)",
                            "Forum JSON (1.31M)",
                            "BoardingPass XLSX (616K)",
                            "Pointz XML (436K)",
                            "Sirena TAB (156K)",
                            "BoardingData CSV (155K)",
                            "Единый пул (4.36M)",
                            "Уникальные рейсы (2.27M)",
                            "Дубликаты (2.09M)",
                            "Профили пассажиров (463.9K)",
                        ],
                    ),
                    link=dict(
                        source=[0, 1, 2, 3, 4, 5, 6, 6, 7],
                        target=[6, 6, 6, 6, 6, 6, 7, 8, 9],
                        value=[
                            1687053,
                            1308987,
                            615771,
                            436090,
                            155757,
                            155147,
                            2266112,
                            2092693,
                            463872,
                        ],
                    ),
                )
            ]
        )
        fig_sankey.update_layout(
            title_text="Трансформация и дедупликация записей",
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_sankey, use_container_width=True)

    with col_sources:
        src_df = pd.DataFrame(
            {
                "Источник": [
                    "SkyTeam-Exchange.yaml",
                    "FrequentFlyerForum-Profiles.json",
                    "YourBoardingPassDotAero/*.xlsx",
                    "PointzAggregator-AirlinesData.xml",
                    "Sirena-export-fixed.tab",
                    "BoardingData.csv",
                ],
                "Записей": [1687053, 1308987, 615771, 436090, 155757, 155147],
                "Специфика обработки": [
                    "Трехуровневый YAML (CSafeLoader)",
                    "Вложенный JSON, профили и списки документов",
                    "Многостраничные книги Excel (read_only)",
                    "XML-дерево активностей, некорректный код REA",
                    "Фиксированная ширина колонок, кириллица",
                    "Даты MM/DD/YYYY, отсутствие кодов аэропортов",
                ],
            }
        )
        st.dataframe(src_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Устраненные дефекты качества данных")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            "**1. Код аэропорта REA (12 601 запись)**\n\n"
            "В `BoardingPass_XLSX` и `Pointz_XML` на рейсах в Сеул (`SU5436`, `KE827`) "
            "вместо кода аэропорта указан фрагмент строки `KOREA` (`REA`). Заменено на `ICN`."
        )
    with c2:
        st.markdown(
            "**2. Инициалы вместо имени в XLSX (61 611 записей)**\n\n"
            "В части посадочных талонов имя сокращено до одной буквы. Для предотвращения ложных "
            "склеек токены длиной менее 2 символов исключены из `full_name_norm`, а связывание "
            "выполнено по номеру билета."
        )
    with c3:
        st.markdown(
            "**3. Пропуски аэропортов и неоднозначность номеров рейсов**\n\n"
            "В `BoardingData.csv` отсутствуют коды аэропортов, при этом 8 172 номера рейсов "
            "используются на нескольких маршрутах. Пропуски заполнены модальными значениями "
            "справочника рейсов."
        )

    st.divider()
    st.subheader("Проверка фонетической нормализации ФИО")
    test_col1, test_col2 = st.columns(2)
    with test_col1:
        inp_name_1 = st.text_input(
            "Запись ФИО (Источник 1):", value="ОЗЕРОВ ИЛЬДАР ДАНИИЛОВИЧ"
        )
        inp_name_2 = st.text_input("Запись ФИО (Источник 2):", value="ILDAR OZEROV")
    with test_col2:
        k1 = transliterate_and_normalize(inp_name_1)
        k2 = transliterate_and_normalize(inp_name_2)
        st.write(f"Ключ 1: `{k1}`")
        st.write(f"Ключ 2: `{k2}`")
        if k1 and k1 == k2:
            st.info("Ключи совпадают.")
        else:
            st.warning("Ключи различаются.")


def render_er_tab():
    st.subheader("Алгоритм связывания профилей (Entity Resolution)")
    er_col1, er_col2 = st.columns(2)

    with er_col1:
        st.markdown(
            "**Этапы построения графа `Union-Find`:**\n"
            "1. Предварительная фильтрация скомпрометированных документов и карт лояльности, "
            "у которых наблюдается более одного `full_name_norm` или `birth_date`.\n"
            "2. Связывание жестких идентификаторов (`DOC`, `TKT`, `FF`, `XUID`, `FUID`).\n"
            "3. Внутрирейсовое объединение по `(flight_date, flight_number, full_name_norm)` "
            "при отсутствии конфликтов по паспортам, датам рождения, инициалам отчества и билетам.\n"
            "4. Межрейсовое объединение по `(full_name_norm, birth_date)` для связки внутреннего "
            "(10 цифр) и заграничного (9 цифр) паспортов.\n"
            "5. Траекторная деанонимизация: объединение анонимного профиля с именованным при "
            "совпадении минимум на 5 рейсах (для списков от 2 до 25 активных пассажиров на борту)."
        )

    with er_col2:
        st.markdown("**Примеры разделения однофамильцев и полных тезок на одном рейсе**")
        namesake_cases = pd.DataFrame(
            [
                {
                    "Дата и рейс": "2017-01-05 | SU4600",
                    "Пассажир 1": "BELIKOVA OKSANA OLEGOVNA",
                    "Паспорт 1": "7561803205",
                    "Пассажир 2": "BELIKOVA OKSANA MATVEEVNA",
                    "Паспорт 2": "9415606085",
                },
                {
                    "Дата и рейс": "2017-01-01 | SU1423",
                    "Пассажир 1": "BULATOV NIKITA VITALEVICH",
                    "Паспорт 1": "3844507206",
                    "Пассажир 2": "BULATOV NIKITA ODISSEICH",
                    "Паспорт 2": "3829844288",
                },
                {
                    "Дата и рейс": "2017-02-22 | SU1212",
                    "Пассажир 1": "GOROKHOVA TAISIIA ARTEMEVNA (1999)",
                    "Паспорт 1": "6007623748",
                    "Пассажир 2": "GOROKHOVA TAISIIA SAMIROVNA (1992)",
                    "Паспорт 2": "9898891869",
                },
            ]
        )
        st.dataframe(namesake_cases, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Примеры деанонимизированных профилей")
    deanon_sample = pd.DataFrame(
        [
            {
                "Анонимный узел": "FUID:FrequentFlyer54887",
                "Установленное ФИО": "LILIIA_VLADIMIROVA",
                "Паспорт": "2709215927",
                "Совпавших рейсов": 31,
            },
            {
                "Анонимный узел": "FUID:FrequentFlyer67981",
                "Установленное ФИО": "IASHIN_KONSTANTIN",
                "Паспорт": "5792409622",
                "Совпавших рейсов": 31,
            },
            {
                "Анонимный узел": "FUID:FrequentFlyer88791",
                "Установленное ФИО": "DANIL_GLEBOV",
                "Паспорт": "217046681",
                "Совпавших рейсов": 31,
            },
            {
                "Анонимный узел": "FUID:FrequentFlyer41586",
                "Установленное ФИО": "ANDREI_HROMOV",
                "Паспорт": "033093080",
                "Совпавших рейсов": 31,
            },
            {
                "Анонимный узел": "FUID:FrequentFlyer65613",
                "Установленное ФИО": "KONDRATEVA_ULIANA",
                "Паспорт": "6269595357",
                "Совпавших рейсов": 16,
            },
            {
                "Анонимный узел": "FF:FB 252689453",
                "Установленное ФИО": "MALCEV_VLAD",
                "Паспорт": "4930958241",
                "Совпавших рейсов": 14,
            },
        ]
    )
    st.dataframe(deanon_sample, use_container_width=True, hide_index=True)


def render_radar_tab(df_summary: pd.DataFrame, df_fl: pd.DataFrame):
    st.subheader("Конфликты идентификаторов (результат автоматической проверки)")
    doc_conflicts, ff_conflicts = get_conflict_tables(df_fl)
    inc_col1, inc_col2 = st.columns(2)

    with inc_col1:
        st.markdown(f"**Конфликтные документы ({len(doc_conflicts)})**")
        st.dataframe(doc_conflicts.reset_index(), use_container_width=True, hide_index=True)
        if not doc_conflicts.empty:
            top_doc = str(doc_conflicts.index[0])
            if st.button(f"Показать перелеты по документу {top_doc}"):
                cols = [
                    "person_id_final", "flight_date", "flight_number",
                    "first_name", "second_name", "last_name", "birth_date",
                    "doc_clean", "ff_card", "ticket_number", "dep_airport", "arr_airport",
                ]
                cols = [c for c in cols if c in df_fl.columns]
                st.dataframe(
                    df_fl[df_fl["doc_clean"] == top_doc][cols],
                    use_container_width=True,
                    hide_index=True,
                )

    with inc_col2:
        st.markdown(f"**Конфликтные карты лояльности ({len(ff_conflicts)})**")
        st.dataframe(ff_conflicts.reset_index(), use_container_width=True, hide_index=True)
        if not ff_conflicts.empty:
            top_ff = str(ff_conflicts.index[0])
            if st.button(f"Показать перелеты по карте {top_ff}"):
                cols = [
                    "person_id_final", "flight_date", "flight_number",
                    "first_name", "last_name", "birth_date",
                    "doc_clean", "ff_card", "dep_airport", "arr_airport",
                ]
                cols = [c for c in cols if c in df_fl.columns]
                st.dataframe(
                    df_fl[df_fl["ff_card"] == top_ff][cols],
                    use_container_width=True,
                    hide_index=True,
                )

    st.divider()
    st.subheader("Параметрический фильтр поведенческих аномалий")
    f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)
    min_flights = f_col1.slider("Мин. число перелетов:", 5, 45, 25)
    min_turnarounds = f_col2.slider("Мин. быстрых разворотов (0-1 сут.):", 0, 10, 4)
    min_nobag_pct = f_col3.slider("Мин. доля без багажа (% от известных):", 0, 100, 50)
    min_known_bag_pct = f_col4.slider("Мин. полнота данных о багаже (%):", 0, 100, 0)
    max_teleports = f_col5.slider("Макс. разрывов маршрута:", 0, 20, 10)

    filtered_suspects = df_summary[
        (df_summary["total_flights"] >= min_flights)
        & (df_summary["quick_turnarounds"] >= min_turnarounds)
        & (df_summary["no_baggage_pct"] >= min_nobag_pct)
        & (df_summary["baggage_known_pct"] >= min_known_bag_pct)
        & (df_summary["teleports"] <= max_teleports)
    ].sort_values(
        ["risk_score", "courier_score", "quick_turnarounds", "total_flights"],
        ascending=False,
    )

    st.write(f"Найдено профилей по заданным критериям: **{len(filtered_suspects)}**")

    sc_col, tbl_col = st.columns([1, 1])
    with sc_col:
        fig_scatter = px.scatter(
            filtered_suspects.head(300),
            x="total_flights",
            y="quick_turnarounds",
            size="unique_airports",
            color="risk_score",
            hover_name="full_name",
            hover_data=[
                "person_id_final",
                "primary_doc",
                "primary_ff_card",
                "no_baggage_pct",
                "baggage_known_pct",
                "teleports",
                "courier_score",
                "identity_score",
            ],
            color_continuous_scale="Viridis",
            labels={
                "total_flights": "Всего перелетов",
                "quick_turnarounds": "Быстрых разворотов (0-1 сут.)",
                "risk_score": "Risk Score",
                "unique_airports": "Уникальных аэропортов",
            },
            title="Распределение кандидатов по частоте перелетов и быстрым разворотам",
        )
        fig_scatter.update_layout(height=420)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tbl_col:
        display_cols = [
            "person_id_final", "full_name", "birth_date", "primary_doc",
            "primary_ff_card", "total_flights", "quick_turnarounds",
            "no_baggage_pct", "baggage_known_pct", "teleports",
            "unique_airports", "identity_score", "courier_score", "risk_score",
        ]
        display_cols = [c for c in display_cols if c in filtered_suspects.columns]
        st.dataframe(
            filtered_suspects[display_cols].head(100),
            use_container_width=True,
            hide_index=True,
            height=420,
        )


def render_dossier_tab(df_summary: pd.DataFrame, df_fl: pd.DataFrame):
    st.subheader("Персональное досье и траектория перемещений")
    search_col1, search_col2 = st.columns([2, 2])

    top_suspects = df_summary.head(15)
    preset_options = [
        f"{row.person_id_final} | {row.full_name} (рейсов: {row.total_flights}, разворотов: {row.quick_turnarounds}, risk: {row.risk_score})"
        for row in top_suspects.itertuples()
    ]

    with search_col1:
        selected_preset = st.selectbox(
            "Выбор профиля из топ-рейтинга по Risk Score:", preset_options
        )
        preset_pid = selected_preset.split(" | ")[0] if selected_preset else ""

    with search_col2:
        custom_query = st.text_input(
            "Поиск по ФИО, паспорту, карте лояльности или ID:",
            placeholder="Например: 8248013778 или FB 139888197 или SVIRIDOVA",
        )

    target_pid = preset_pid
    if custom_query.strip():
        q = custom_query.strip().upper()
        matches = df_summary[
            df_summary["person_id_final"].astype(str).str.upper().str.contains(q, na=False)
            | df_summary["full_name"].astype(str).str.upper().str.contains(q, na=False)
            | df_summary["primary_doc"].astype(str).str.upper().str.contains(q, na=False)
            | df_summary["primary_ff_card"].astype(str).str.upper().str.contains(q, na=False)
        ]
        if not matches.empty:
            match_labels = (
                matches["person_id_final"].astype(str)
                + " | "
                + matches["full_name"].fillna("ANONYMOUS").astype(str)
                + " ("
                + matches["total_flights"].astype(str)
                + " рейсов)"
            )
            selected_match = st.selectbox(
                f"Найдено совпадений ({len(matches)}):", match_labels
            )
            target_pid = selected_match.split(" | ")[0]
        else:
            st.warning("Совпадений не найдено. Отображается профиль из списка слева.")

    pax_row = df_summary[df_summary["person_id_final"] == target_pid]
    pax_flights = (
        df_fl[df_fl["person_id_final"] == target_pid]
        .sort_values(["flight_date", "depart_time"])
        .reset_index(drop=True)
    )

    if not pax_row.empty:
        pr = pax_row.iloc[0]
        d1, d2, d3, d4, d5, d6 = st.columns(6)
        d1.metric("ФИО", str(pr["full_name"]))
        d2.metric("Паспорт", str(pr["primary_doc"]))
        d3.metric("Карта лояльности", str(pr["primary_ff_card"]))
        d4.metric(
            "Всего перелетов",
            int(pr["total_flights"]),
            f"Аэропортов: {int(pr['unique_airports'])}",
        )
        d5.metric(
            "Быстрых разворотов",
            int(pr["quick_turnarounds"]),
            f"Разрывов: {int(pr['teleports'])}",
        )
        d6.metric(
            "Без багажа (% изв.)",
            f"{pr['no_baggage_pct']}% (изв. {pr['baggage_known_pct']}%)",
            f"Risk Score: {pr['risk_score']}",
        )

    map_col, chart_col = st.columns([3, 2])

    with map_col:
        fig_map = go.Figure()
        visited_airports = set()

        for _, fl in pax_flights.iterrows():
            dep, arr = fl.get("dep_airport"), fl.get("arr_airport")
            if dep in AIRPORT_COORDS and arr in AIRPORT_COORDS:
                lat1, lon1, _ = AIRPORT_COORDS[dep]
                lat2, lon2, _ = AIRPORT_COORDS[arr]
                visited_airports.add(dep)
                visited_airports.add(arr)
                is_turn = bool(fl.get("is_quick_turnaround", False))
                turn_label = "Быстрый разворот" if is_turn else "Стандартный рейс"
                fig_map.add_trace(
                    go.Scattergeo(
                        lon=[lon1, lon2],
                        lat=[lat1, lat2],
                        mode="lines",
                        line=dict(
                            width=2.2 if is_turn else 1.4,
                            color="#d62728" if is_turn else "#1f77b4",
                        ),
                        opacity=0.8,
                        hoverinfo="text",
                        text=f"{fl['flight_date']} | {fl['flight_number']}: {dep} -> {arr} ({turn_label})",
                        showlegend=False,
                    )
                )

        if visited_airports:
            ap_list = sorted(visited_airports)
            fig_map.add_trace(
                go.Scattergeo(
                    lon=[AIRPORT_COORDS[a][1] for a in ap_list],
                    lat=[AIRPORT_COORDS[a][0] for a in ap_list],
                    mode="markers+text",
                    text=ap_list,
                    textposition="top center",
                    marker=dict(
                        size=7, color="#ff7f0e", line=dict(width=1, color="black")
                    ),
                    hovertext=[f"{a} ({AIRPORT_COORDS[a][2]})" for a in ap_list],
                    name="Аэропорты",
                )
            )

        fig_map.update_layout(
            title="Карта маршрутов (красный цвет — разворот за 0-1 сутки)",
            height=430,
            margin=dict(l=0, r=0, t=40, b=0),
            geo=dict(
                projection_type="natural earth",
                showland=True,
                landcolor="rgb(243, 243, 243)",
                countrycolor="rgb(204, 204, 204)",
                showcountries=True,
            ),
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with chart_col:
        if not pax_flights.empty:
            pax_flights["month"] = (
                pax_flights["flight_date"].astype(str).str.slice(0, 7)
            )
            m_counts = pax_flights.groupby("month").size().reset_index(name="flights")
            fig_bar = px.bar(
                m_counts,
                x="month",
                y="flights",
                title="Динамика перелетов по месяцам",
                labels={"month": "Месяц", "flights": "Количество рейсов"},
            )
            fig_bar.update_layout(height=430, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader(f"Ведомость перелетов профиля {target_pid} ({len(pax_flights)})")
    show_cols = [
        "flight_date", "depart_time", "flight_number", "dep_airport", "arr_airport",
        "dep_city", "dest_city", "first_name", "second_name", "last_name",
        "doc_clean", "ff_card", "ticket_number", "booking_code",
        "travel_class", "fare", "baggage", "no_baggage",
        "is_quick_turnaround", "is_teleport",
    ]
    show_cols = [c for c in show_cols if c in pax_flights.columns]
    st.dataframe(pax_flights[show_cols], use_container_width=True, hide_index=True)


def main():
    st.set_page_config(
        page_title="Yoyoflot Security & ETL Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    df_summary = load_summary()
    df_flights = load_flights()
    doc_conflicts, ff_conflicts = get_conflict_tables(df_flights)

    st.title("Yoyoflot Airlines — Система интеграции данных и анализа аномалий")
    st.markdown(
        "Аналитическая панель результатов конвейера обработки данных (6 источников), "
        "графового связывания профилей и выявления аномальных паттернов перелетов."
    )

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Сырых записей", "4 358 805", "6 источников")
    m2.metric("Уникальных перелетов", f"{len(df_flights):,}".replace(",", " "))
    m3.metric("Профилей пассажиров", f"{len(df_summary):,}".replace(",", " "))
    m4.metric("Конфликтных паспортов", str(len(doc_conflicts)))
    m5.metric("Конфликтных карт", str(len(ff_conflicts)))
    m6.metric(
        "Профилей с конфликтами ID",
        str(int((df_summary["identity_score"] > 0).sum())),
    )

    st.divider()

    tab_etl, tab_er, tab_radar, tab_dossier = st.tabs(
        [
            "Этап 1: ETL и очистка",
            "Этап 2: Entity Resolution",
            "Этап 3: Аномалии и конфликты",
            "Этап 4: Досье пассажира",
        ]
    )

    with tab_etl:
        render_etl_tab()
    with tab_er:
        render_er_tab()
    with tab_radar:
        render_radar_tab(df_summary, df_flights)
    with tab_dossier:
        render_dossier_tab(df_summary, df_flights)


if __name__ == "__main__":
    main()
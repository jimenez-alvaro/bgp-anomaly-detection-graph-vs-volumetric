from BML.data import Dataset
from BML.transform import DatasetTransformation
from BML import utils
import os


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def make_event(name, label, year, month, day, hour_start, hour_end):
    """
    Build a BML period-of-interest dictionary for a single incident window.

    Parameters
    ----------
    name        : str   Unique identifier for the event (e.g. 'YouTube_Hijack')
    label       : str   'normal' for baseline traffic, 'ataque' for anomaly
    year        : int
    month       : int
    day         : int
    hour_start  : int   Start hour (UTC) of the 3-hour observation window
    hour_end    : int   End hour (UTC) of the 3-hour observation window
    """
    return {
        "name":       name,
        "label":      label,
        "start_time": utils.getTimestamp(year, month, day, hour_start, 0, 0),
        "end_time":   utils.getTimestamp(year, month, day, hour_end,   0, 0),
    }


def extract_category(folder, events):
    """
    Run the full BML extraction pipeline for a given anomaly category.

    Steps:
        1. Download raw BGP data (RIBs + Updates) from RIPE RIS rrc04
        2. Extract 32 volumetric features at 5-minute resolution
        3. Extract 14 topological graph features at 5-minute resolution
           (with nbProcess=1 to prevent silent memory overflow in BML)

    Parameters
    ----------
    folder : str   Output directory for this category
    events : list  List of event dicts produced by make_event()
    """
    os.makedirs(folder, exist_ok=True)

    dataset = Dataset(folder)
    dataset.setParams({
        "PrimingPeriod": 2 * 60 * 60,   # 2-hour priming window before each event
        "IpVersion":     [4],            # IPv4 only
        "Collectors":    ["rrc04"],      # RIPE RIS Geneva collector
    })
    dataset.setPeriodsOfInterests(events)

    # --- Step 1: Raw data download ---
    print(f"  [1/3] Downloading raw BGP data...")
    utils.runJobs(dataset.getJobs(), os.path.join(folder, "collect_jobs"))

    # --- Step 2: Volumetric features ---
    print(f"  [2/3] Extracting volumetric features (32 features, 5-min resolution)...")
    vol = DatasetTransformation(folder, "BML.transform", "Features")
    vol.setParams({"global": {"Period": 5}})
    utils.runJobs(vol.getJobs(), os.path.join(folder, "transform_jobs_features"))

    # --- Step 3: Topological graph features ---
    # nbProcess=1 and nbProcessFeatures=1 disable BML's internal multiprocessing
    # pool to prevent silent memory overflow when building graph metrics.
    print(f"  [3/3] Extracting topological graph features (14 features, 5-min resolution)...")
    graph = DatasetTransformation(folder, "BML.transform", "GraphFeatures")
    graph.setParams({
        "global": {
            "Period":            5,
            "nbProcess":         1,
            "nbProcessFeatures": 1,
        }
    })
    utils.runJobs(graph.getJobs(), os.path.join(folder, "transform_jobs_graphfeatures"))

    print(f"  Done. Output saved to: {folder}\n")


# ==============================================================================
# INCIDENT DEFINITIONS
# ==============================================================================

ORIGIN_HIJACKS = [
    make_event("YouTube_Normal",           "normal", 2008,  2, 23, 18, 21),
    make_event("YouTube_Hijack",           "ataque", 2008,  2, 24, 18, 21),
    make_event("Amazon_MEW_Normal",        "normal", 2018,  4, 23, 11, 14),
    make_event("Amazon_MEW_Hijack",        "ataque", 2018,  4, 24, 11, 14),
    make_event("Rostelecom_Normal",        "normal", 2020,  3, 31, 19, 22),
    make_event("Rostelecom_Hijack",        "ataque", 2020,  4,  1, 19, 22),
    make_event("Twitter_Normal",           "normal", 2022,  2, 27, 10, 13),
    make_event("Twitter_Hijack",           "ataque", 2022,  2, 28, 10, 13),
    make_event("CelerNetwork_Normal",      "normal", 2022,  8, 16, 19, 22),
    make_event("CelerNetwork_Hijack",      "ataque", 2022,  8, 17, 19, 22),
    make_event("ChinaTelecom2010_Normal",  "normal", 2010,  4,  7, 14, 17),
    make_event("ChinaTelecom2010_Hijack",  "ataque", 2010,  4,  8, 14, 17),
    make_event("Beltelecom2013_Normal",    "normal", 2013,  2, 19, 10, 13),
    make_event("Beltelecom2013_Hijack",    "ataque", 2013,  2, 20, 10, 13),
    make_event("HackingTeam2013_Normal",   "normal", 2013,  8,  6, 12, 15),
    make_event("HackingTeam2013_Hijack",   "ataque", 2013,  8,  7, 12, 15),
    make_event("Indosat2014_Normal",       "normal", 2014,  4,  1, 10, 13),
    make_event("Indosat2014_Hijack",       "ataque", 2014,  4,  2, 10, 13),
    make_event("Airtel2015_Normal",        "normal", 2015, 11, 11, 14, 17),
    make_event("Airtel2015_Hijack",        "ataque", 2015, 11, 12, 14, 17),
    make_event("eNet_Google2018_Normal",   "normal", 2018, 11, 11, 12, 15),
    make_event("eNet_Google2018_Hijack",   "ataque", 2018, 11, 12, 12, 15),
    make_event("Telstra2019_Normal",       "normal", 2019,  7, 10,  8, 11),
    make_event("Telstra2019_Hijack",       "ataque", 2019,  7, 11,  8, 11),
    make_event("Lumen2020_Normal",         "normal", 2020,  8, 29, 16, 19),
    make_event("Lumen2020_Hijack",         "ataque", 2020,  8, 30, 16, 19),
    make_event("AS209306_2021_Normal",     "normal", 2021,  3, 21, 10, 13),
    make_event("AS209306_2021_Hijack",     "ataque", 2021,  3, 22, 10, 13),
    make_event("VodafoneAS55410_Normal",   "normal", 2021,  4, 20, 12, 15),
    make_event("VodafoneAS55410_Hijack",   "ataque", 2021,  4, 21, 12, 15),
    make_event("CERNET2021_Normal",        "normal", 2021,  6,  6, 10, 13),
    make_event("CERNET2021_Hijack",        "ataque", 2021,  6,  7, 10, 13),
    make_event("TurkTelecom2022_Normal",   "normal", 2022,  3, 10,  9, 12),
    make_event("TurkTelecom2022_Hijack",   "ataque", 2022,  3, 11,  9, 12),
    make_event("AS267613_2022_Normal",     "normal", 2022,  8,  3, 14, 17),
    make_event("AS267613_2022_Hijack",     "ataque", 2022,  8,  4, 14, 17),
    make_event("MangoDSL2022_Normal",      "normal", 2022, 10, 18, 10, 13),
    make_event("MangoDSL2022_Hijack",      "ataque", 2022, 10, 19, 10, 13),
    make_event("Cobranet2023_Normal",      "normal", 2023,  1, 23, 10, 13),
    make_event("Cobranet2023_Hijack",      "ataque", 2023,  1, 24, 10, 13),
    make_event("Eletronet2023_Normal",     "normal", 2023,  3, 14, 12, 15),
    make_event("Eletronet2023_Hijack",     "ataque", 2023,  3, 15, 12, 15),
    make_event("MyRepublic2023_Normal",    "normal", 2023,  5,  9, 10, 13),
    make_event("MyRepublic2023_Hijack",    "ataque", 2023,  5, 10, 10, 13),
    make_event("RCSRDS2023_Normal",        "normal", 2023,  7, 17, 10, 13),
    make_event("RCSRDS2023_Hijack",        "ataque", 2023,  7, 18, 10, 13),
    make_event("Digicel2023_Normal",       "normal", 2023,  9, 19, 14, 17),
    make_event("Digicel2023_Hijack",       "ataque", 2023,  9, 20, 14, 17),
    make_event("UzbekAS2023_Normal",       "normal", 2023, 10,  3, 10, 13),
    make_event("UzbekAS2023_Hijack",       "ataque", 2023, 10,  4, 10, 13),
    make_event("PTCL2023_Normal",          "normal", 2023, 11, 14, 12, 15),
    make_event("PTCL2023_Hijack",          "ataque", 2023, 11, 15, 12, 15),
]

PATH_HIJACKS = [
    make_event("Bitcoin_AS32653_Normal",    "normal", 2014,  8,  6, 14, 17),
    make_event("Bitcoin_AS32653_Hijack",    "ataque", 2014,  8,  7, 14, 17),
    make_event("Visa_Rostelecom_Normal",    "normal", 2017, 12, 11, 10, 13),
    make_event("Visa_Rostelecom_Hijack",    "ataque", 2017, 12, 12, 10, 13),
    make_event("Bitcanal_Normal",           "normal", 2018,  7,  8, 10, 13),
    make_event("Bitcanal_Hijack",           "ataque", 2018,  7,  9, 10, 13),
    make_event("ChinaTelecom_Normal",       "normal", 2019,  6,  5, 10, 13),
    make_event("ChinaTelecom_Hijack",       "ataque", 2019,  6,  6, 10, 13),
    make_event("Swisscom_Normal",           "normal", 2019, 12,  8,  9, 12),
    make_event("Swisscom_Hijack",           "ataque", 2019, 12,  9,  9, 12),
    make_event("Renesys2008_Normal",        "normal", 2008,  3, 23, 10, 13),
    make_event("Renesys2008_Hijack",        "ataque", 2008,  3, 24, 10, 13),
    make_event("GlobalCrossing2010_Normal", "normal", 2010,  9, 14, 12, 15),
    make_event("GlobalCrossing2010_Hijack", "ataque", 2010,  9, 15, 12, 15),
    make_event("TelecomItalia2011_Normal",  "normal", 2011,  5, 18, 10, 13),
    make_event("TelecomItalia2011_Hijack",  "ataque", 2011,  5, 19, 10, 13),
    make_event("Rostelecom2017_Normal",     "normal", 2017,  4, 26, 10, 13),
    make_event("Rostelecom2017_Hijack",     "ataque", 2017,  4, 27, 10, 13),
    make_event("Level3_2017_Normal",        "normal", 2017, 11,  5, 10, 13),
    make_event("Level3_2017_Hijack",        "ataque", 2017, 11,  6, 10, 13),
    make_event("Backconnect2018_Normal",    "normal", 2018,  2, 11, 12, 15),
    make_event("Backconnect2018_Hijack",    "ataque", 2018,  2, 12, 12, 15),
    make_event("EthereumClassic2019_Normal","normal", 2019,  1,  4, 14, 17),
    make_event("EthereumClassic2019_Hijack","ataque", 2019,  1,  5, 14, 17),
    make_event("Maxis2019_Normal",          "normal", 2019,  6, 23, 10, 13),
    make_event("Maxis2019_Hijack",          "ataque", 2019,  6, 24, 10, 13),
    make_event("Telekomunikacja2019_Normal","normal", 2019,  9, 16, 10, 13),
    make_event("Telekomunikacja2019_Hijack","ataque", 2019,  9, 17, 10, 13),
    make_event("M247_2020_Normal",          "normal", 2020,  2, 18, 10, 13),
    make_event("M247_2020_Hijack",          "ataque", 2020,  2, 19, 10, 13),
    make_event("Windstream2020_Normal",     "normal", 2020,  6, 14, 12, 15),
    make_event("Windstream2020_Hijack",     "ataque", 2020,  6, 15, 12, 15),
    make_event("TATA2021_Normal",           "normal", 2021,  3,  8, 10, 13),
    make_event("TATA2021_Hijack",           "ataque", 2021,  3,  9, 10, 13),
    make_event("Zayo2021_Normal",           "normal", 2021,  7, 12, 10, 13),
    make_event("Zayo2021_Hijack",           "ataque", 2021,  7, 13, 10, 13),
    make_event("NTT2021_Normal",            "normal", 2021, 10, 12, 10, 13),
    make_event("NTT2021_Hijack",            "ataque", 2021, 10, 13, 10, 13),
    make_event("Rostelecom2022_Normal",     "normal", 2022,  3, 28, 10, 13),
    make_event("Rostelecom2022_Hijack",     "ataque", 2022,  3, 29, 10, 13),
    make_event("Cogent2022_Normal",         "normal", 2022,  6,  7, 10, 13),
    make_event("Cogent2022_Hijack",         "ataque", 2022,  6,  8, 10, 13),
    make_event("ChinaMobile2022_Normal",    "normal", 2022,  9, 20, 10, 13),
    make_event("ChinaMobile2022_Hijack",    "ataque", 2022,  9, 21, 10, 13),
    make_event("Turkcell2022_Normal",       "normal", 2022, 11, 15, 10, 13),
    make_event("Turkcell2022_Hijack",       "ataque", 2022, 11, 16, 10, 13),
    make_event("Bharti2023_Normal",         "normal", 2023,  1, 16, 10, 13),
    make_event("Bharti2023_Hijack",         "ataque", 2023,  1, 17, 10, 13),
    make_event("RETN2023_Normal",           "normal", 2023,  4, 10, 10, 13),
    make_event("RETN2023_Hijack",           "ataque", 2023,  4, 11, 10, 13),
    make_event("Zayo2023_Normal",           "normal", 2023,  6, 19, 10, 13),
    make_event("Zayo2023_Hijack",           "ataque", 2023,  6, 20, 10, 13),
    make_event("Lumen2023_Normal",          "normal", 2023,  8, 21, 10, 13),
    make_event("Lumen2023_Hijack",          "ataque", 2023,  8, 22, 10, 13),
    make_event("MTS2023_Normal",            "normal", 2023, 10, 16, 10, 13),
    make_event("MTS2023_Hijack",            "ataque", 2023, 10, 17, 10, 13),
]

ROUTE_LEAKS = [
    make_event("TelekomMalaysia_Normal",    "normal", 2015,  6, 11,  8, 11),
    make_event("TelekomMalaysia_Leak",      "ataque", 2015,  6, 12,  8, 11),
    make_event("Google_MainOne_Normal",     "normal", 2018, 11, 11, 20, 23),
    make_event("Google_MainOne_Leak",       "ataque", 2018, 11, 12, 20, 23),
    make_event("Cloudflare_Normal",         "normal", 2019,  6, 23, 10, 13),
    make_event("Cloudflare_Leak",           "ataque", 2019,  6, 24, 10, 13),
    make_event("CenturyLink_Normal",        "normal", 2020,  8, 29, 10, 13),
    make_event("CenturyLink_Leak",          "ataque", 2020,  8, 30, 10, 13),
    make_event("Vodafone_India_Normal",     "normal", 2021,  4, 15, 13, 16),
    make_event("Vodafone_India_Leak",       "ataque", 2021,  4, 16, 13, 16),
    make_event("TurkTelecom2004_Normal",    "normal", 2004, 12, 23, 12, 15),
    make_event("TurkTelecom2004_Leak",      "ataque", 2004, 12, 24, 12, 15),
    make_event("AGIS2010_Normal",           "normal", 2010,  1, 11, 10, 13),
    make_event("AGIS2010_Leak",             "ataque", 2010,  1, 12, 10, 13),
    make_event("ChinaTelecom2010Leak_Normal","normal",2010,  4,  7, 10, 13),
    make_event("ChinaTelecom2010Leak_Leak", "ataque", 2010,  4,  8, 10, 13),
    make_event("Moratel2012_Normal",        "normal", 2012, 11,  5, 12, 15),
    make_event("Moratel2012_Leak",          "ataque", 2012, 11,  6, 12, 15),
    make_event("Dodo2014_Normal",           "normal", 2014,  2, 10, 10, 13),
    make_event("Dodo2014_Leak",             "ataque", 2014,  2, 11, 10, 13),
    make_event("Hathway2015_Normal",        "normal", 2015,  9, 14, 10, 13),
    make_event("Hathway2015_Leak",          "ataque", 2015,  9, 15, 10, 13),
    make_event("Telstra2016_Normal",        "normal", 2016, 11, 14, 10, 13),
    make_event("Telstra2016_Leak",          "ataque", 2016, 11, 15, 10, 13),
    make_event("Level3_2017_Leak_Normal",   "normal", 2017, 11,  5, 14, 17),
    make_event("Level3_2017_Leak",          "ataque", 2017, 11,  6, 14, 17),
    make_event("Brazil2017_Normal",         "normal", 2017, 10, 10, 10, 13),
    make_event("Brazil2017_Leak",           "ataque", 2017, 10, 11, 10, 13),
    make_event("Japan2017_Normal",          "normal", 2017,  8, 24, 10, 13),
    make_event("Japan2017_Leak",            "ataque", 2017,  8, 25, 10, 13),
    make_event("Verizon2019_Normal",        "normal", 2019,  6, 23, 14, 17),
    make_event("Verizon2019_Leak",          "ataque", 2019,  6, 24, 14, 17),
    make_event("Telus2020_Normal",          "normal", 2020,  7, 21, 10, 13),
    make_event("Telus2020_Leak",            "ataque", 2020,  7, 22, 10, 13),
    make_event("CoreBackbone2020_Normal",   "normal", 2020,  9,  8, 10, 13),
    make_event("CoreBackbone2020_Leak",     "ataque", 2020,  9,  9, 10, 13),
    make_event("Iliad2021_Normal",          "normal", 2021,  2, 15, 10, 13),
    make_event("Iliad2021_Leak",            "ataque", 2021,  2, 16, 10, 13),
    make_event("TelekomSerbia2021_Normal",  "normal", 2021,  6, 14, 10, 13),
    make_event("TelekomSerbia2021_Leak",    "ataque", 2021,  6, 15, 10, 13),
    make_event("Guangdong2021_Normal",      "normal", 2021,  8, 23, 10, 13),
    make_event("Guangdong2021_Leak",        "ataque", 2021,  8, 24, 10, 13),
    make_event("Telstra2022_Normal",        "normal", 2022,  1, 17, 10, 13),
    make_event("Telstra2022_Leak",          "ataque", 2022,  1, 18, 10, 13),
    make_event("Altice2022_Normal",         "normal", 2022,  3, 21, 10, 13),
    make_event("Altice2022_Leak",           "ataque", 2022,  3, 22, 10, 13),
    make_event("Jio2022_Normal",            "normal", 2022,  7, 18, 10, 13),
    make_event("Jio2022_Leak",              "ataque", 2022,  7, 19, 10, 13),
    make_event("STC2022_Normal",            "normal", 2022, 10, 24, 10, 13),
    make_event("STC2022_Leak",              "ataque", 2022, 10, 25, 10, 13),
    make_event("TurkTelecom2022Leak_Normal","normal", 2022, 11, 21, 10, 13),
    make_event("TurkTelecom2022Leak_Leak",  "ataque", 2022, 11, 22, 10, 13),
    make_event("BSNL2023_Normal",           "normal", 2023,  2, 13, 10, 13),
    make_event("BSNL2023_Leak",             "ataque", 2023,  2, 14, 10, 13),
    make_event("Rostelecom2023Leak_Normal", "normal", 2023,  4, 17, 10, 13),
    make_event("Rostelecom2023Leak_Leak",   "ataque", 2023,  4, 18, 10, 13),
    make_event("OrangeSpain2023_Normal",    "normal", 2023,  7, 10, 10, 13),
    make_event("OrangeSpain2023_Leak",      "ataque", 2023,  7, 11, 10, 13),
    make_event("Telia2023_Normal",          "normal", 2023, 10, 23, 10, 13),
    make_event("Telia2023_Leak",            "ataque", 2023, 10, 24, 10, 13),
]

MASS_OUTAGES = [
    make_event("Facebook_Normal",           "normal", 2021, 10,  3, 15, 18),
    make_event("Facebook_Outage",           "ataque", 2021, 10,  4, 15, 18),
    make_event("Rogers_Canada_Normal",      "normal", 2022,  7,  7,  8, 11),
    make_event("Rogers_Canada_Outage",      "ataque", 2022,  7,  8,  8, 11),
    make_event("KDDI_Japan_Normal",         "normal", 2022,  6, 30, 16, 19),
    make_event("KDDI_Japan_Outage",         "ataque", 2022,  7,  1, 16, 19),
    make_event("Spark_NZ_Normal",           "normal", 2023,  9,  5, 14, 17),
    make_event("Spark_NZ_Outage",           "ataque", 2023,  9,  6, 14, 17),
    make_event("Optus_Aus_Normal",          "normal", 2023, 11,  7, 17, 20),
    make_event("Optus_Aus_Outage",          "ataque", 2023, 11,  8, 17, 20),
    make_event("Pakistan2011_Normal",       "normal", 2011,  1, 26, 10, 13),
    make_event("Pakistan2011_Outage",       "ataque", 2011,  1, 27, 10, 13),
    make_event("HurricaneSandy2012_Normal", "normal", 2012, 10, 28, 14, 17),
    make_event("HurricaneSandy2012_Outage", "ataque", 2012, 10, 29, 14, 17),
    make_event("Syria2013_Normal",          "normal", 2013,  5,  6, 10, 13),
    make_event("Syria2013_Outage",          "ataque", 2013,  5,  7, 10, 13),
    make_event("Turkey2014_Normal",         "normal", 2014,  3, 26, 20, 23),
    make_event("Turkey2014_Outage",         "ataque", 2014,  3, 27, 20, 23),
    make_event("Libya2014_Normal",          "normal", 2014,  6, 12, 10, 13),
    make_event("Libya2014_Outage",          "ataque", 2014,  6, 13, 10, 13),
    make_event("BT2015_Normal",             "normal", 2015,  6, 18, 10, 13),
    make_event("BT2015_Outage",             "ataque", 2015,  6, 19, 10, 13),
    make_event("Telia2016_Normal",          "normal", 2016,  6, 11, 10, 13),
    make_event("Telia2016_Outage",          "ataque", 2016,  6, 12, 10, 13),
    make_event("DYN2016_Normal",            "normal", 2016, 10, 20, 12, 15),
    make_event("DYN2016_Outage",            "ataque", 2016, 10, 21, 12, 15),
    make_event("Comcast2017_Normal",        "normal", 2017,  9, 19, 10, 13),
    make_event("Comcast2017_Outage",        "ataque", 2017,  9, 20, 10, 13),
    make_event("Iran2019_Normal",           "normal", 2019, 11, 15, 16, 19),
    make_event("Iran2019_Outage",           "ataque", 2019, 11, 16, 16, 19),
    make_event("Cloudflare2020_Normal",     "normal", 2020,  7, 16, 18, 21),
    make_event("Cloudflare2020_Outage",     "ataque", 2020,  7, 17, 18, 21),
    make_event("Google2020_Normal",         "normal", 2020, 11, 15, 14, 17),
    make_event("Google2020_Outage",         "ataque", 2020, 11, 16, 14, 17),
    make_event("Fastly2021_Normal",         "normal", 2021,  6,  7,  9, 12),
    make_event("Fastly2021_Outage",         "ataque", 2021,  6,  8,  9, 12),
    make_event("Akamai2021_Normal",         "normal", 2021,  7, 21, 16, 19),
    make_event("Akamai2021_Outage",         "ataque", 2021,  7, 22, 16, 19),
    make_event("AWS2021_Normal",            "normal", 2021, 12,  6, 10, 13),
    make_event("AWS2021_Outage",            "ataque", 2021, 12,  7, 10, 13),
    make_event("Lumen2022_Normal",          "normal", 2022,  1, 13, 10, 13),
    make_event("Lumen2022_Outage",          "ataque", 2022,  1, 14, 10, 13),
    make_event("Slack2022_Normal",          "normal", 2022,  2, 21, 16, 19),
    make_event("Slack2022_Outage",          "ataque", 2022,  2, 22, 16, 19),
    make_event("Cloudflare2022_Normal",     "normal", 2022,  6, 20, 14, 17),
    make_event("Cloudflare2022_Outage",     "ataque", 2022,  6, 21, 14, 17),
    make_event("Azure2022_Normal",          "normal", 2022,  7, 18, 14, 17),
    make_event("Azure2022_Outage",          "ataque", 2022,  7, 19, 14, 17),
    make_event("Iran2022_Normal",           "normal", 2022,  9, 21, 16, 19),
    make_event("Iran2022_Outage",           "ataque", 2022,  9, 22, 16, 19),
    make_event("Telstra2023_Normal",        "normal", 2023,  2,  7, 10, 13),
    make_event("Telstra2023_Outage",        "ataque", 2023,  2,  8, 10, 13),
    make_event("OrangeSpain2023Out_Normal", "normal", 2023,  1,  2, 10, 13),
    make_event("OrangeSpain2023Out_Outage", "ataque", 2023,  1,  3, 10, 13),
    make_event("VodafoneUK2023_Normal",     "normal", 2023,  6, 11, 10, 13),
    make_event("VodafoneUK2023_Outage",     "ataque", 2023,  6, 12, 10, 13),
    make_event("Swisscom2023_Normal",       "normal", 2023,  8, 14, 10, 13),
    make_event("Swisscom2023_Outage",       "ataque", 2023,  8, 15, 10, 13),
    make_event("NTT2023_Normal",            "normal", 2023, 10,  9, 10, 13),
    make_event("NTT2023_Outage",            "ataque", 2023, 10, 10, 10, 13),
]


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":

    categories = [
        ("dataset_grande/origin_hijacks/", ORIGIN_HIJACKS),
        ("dataset_grande/path_hijacks/",   PATH_HIJACKS),
        ("dataset_grande/route_leaks/",    ROUTE_LEAKS),
        ("dataset_grande/mass_outages/",   MASS_OUTAGES),
    ]

    for folder, events in categories:
        print(f"\n{'='*60}")
        print(f"Processing: {folder}  ({len(events)//2} incidents)")
        print(f"{'='*60}")
        extract_category(folder, events)

    print("All categories extracted successfully.")
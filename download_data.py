from starstream import DataDownloading, WIND
from utils import create_dates

if __name__ == "__main__":
    for tuple_date in create_dates():
        try:
            DataDownloading(
                [
                    WIND.MAG(root="/data/Vlasov/Electron"),
                    WIND.SWE_electron_moments(root="/data/Vlasov/Electron"),
                    WIND.SWE_Ion_Anistropy(root="/data/Vlasov/Proton_anistropy"),
                ],
                tuple_date,
            )
        except:
            continue

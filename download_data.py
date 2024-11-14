from starstream import DataDownloading, WIND, DSCOVR, ACE
from utils import create_dates

if __name__ == "__main__":
    for tuple_date in create_dates():
        try:
            DataDownloading(
                [
                    WIND.TDP_PM(root="/data/Vlasov/TDP_PM"),
                    DSCOVR.FaradayCup(root="/data/Vlasov/DSCOVR/FaradayCup"),
                    DSCOVR.Magnetometer(root="/data/Vlasov/DSCOVR/Magnetometer"),
                    ACE.SWEPAM(download_path="/data/Vlasov/ACE/SWEPAM"),
                    ACE.SWICS(download_path="/data/Vlasov/ACE/SWICS"),
                    ACE.EPAM(download_path="/data/Vlasov/ACE/EPAM"),
                    ACE.MAG(download_path="/data/Vlasov/ACE/MAG"),
                    ACE.SIS(download_path="/data/Vlasov/ACE/SIS"),
                ],
                tuple_date,
            )
        except:
            continue

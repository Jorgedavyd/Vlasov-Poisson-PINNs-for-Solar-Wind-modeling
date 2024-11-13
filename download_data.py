from starstream import DataDownloading, WIND, DSCOVR
from utils import create_dates

if __name__ == "__main__":
    DataDownloading(
        [
            WIND.TDP_PM(root="/data/Vlasov/TDP_PM"),
            DSCOVR.FaradayCup(root="/data/Vlasov/DSCOVR"),
            DSCOVR.Magnetometer(root="/data/Vlasov/DSCOVR"),
        ],
        create_dates(),
    )

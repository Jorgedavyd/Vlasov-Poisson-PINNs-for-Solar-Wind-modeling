from starstream import DataDownloading, WIND, DSCOVR
from utils import create_dates

if __name__ == "__main__":
    DataDownloading(
        [
            WIND.TDP_PM(root="/data/Vlasov"),
            # DSCOVR.FaradayCup(root = '/data/Vlasov'),
            # DSCOVR.Magnetometer(root =  '/data/Vlasov'),
        ],
        create_dates(),
    )

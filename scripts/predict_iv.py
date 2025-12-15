from core.option_price_predictor import predict_all_next_prices
from utils.dataloader import load_data


def main():
    snapshot, expiration_date = load_data()
    predict_all_next_prices(snapshot, expiration_date, cache=True)


if __name__ == "__main__":
    main()

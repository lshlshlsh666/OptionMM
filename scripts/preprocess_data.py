from core.implied_vol_calculator import estimate_iv, estimate_rate
from utils.dataloader import load_raw_data


def main():
    snapshot, expiration_date = load_raw_data()
    r = estimate_rate(snapshot, expiration_date, cache=True)
    estimate_iv(snapshot, expiration_date, r, binomial_tree_steps=100, n_workers=87, cache=True)


if __name__ == "__main__":
    main()

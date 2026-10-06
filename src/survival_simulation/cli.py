import argparse
from pathlib import Path
import pandas as pd
from survival_simulation.simulation import simulate_survival


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic survival data.")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--n-features", type=int, default=50, help="Number of features")
    parser.add_argument("--scenario", type=str, default="linear", choices=["linear", "nonlinear", "non_ph"])
    parser.add_argument("--noise", type=float, default=0.0, help="Additive noise std")
    parser.add_argument("--censoring", type=float, default=0.3, help="Censoring rate [0, 1)")
    parser.add_argument("--correlation", type=float, default=0.0, help="Feature correlation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, required=True, help="Output file path (.csv or .parquet)")

    args = parser.parse_args()

    X, y_censored, _ = simulate_survival(
        n_samples=args.n_samples,
        n_features=args.n_features,
        scenario=args.scenario,
        noise=args.noise,
        censoring=args.censoring,
        correlation=args.correlation,
        seed=args.seed,
        return_type="pandas",
    )

    feature_cols = [f"X{i}" for i in range(args.n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df["event"] = y_censored["event"]
    df["time"] = y_censored["time"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    print(f"Successfully generated {args.n_samples} samples -> {out_path}")


if __name__ == "__main__":
    main()
from gemla.data import make_market_microstructure
from gemla.pipelines import GemlaPipeline
from gemla.reports import export_markdown_report


def main() -> None:
    X = make_market_microstructure()

    pipe = GemlaPipeline()
    result = pipe.fit_evaluate(X)

    print("GEMLA Market Microstructure Demo")
    print("--------------------------------")
    print(result.summary())

    report_path = export_markdown_report(
        result,
        output_path="reports/market_microstructure_report.md",
        title="GEMLA Market Microstructure Report",
    )

    print()
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
from gemla.data import make_synthetic_transport
from gemla.pipelines import GemlaPipeline
from gemla.reports import export_markdown_report


def main() -> None:
    X = make_synthetic_transport()

    pipe = GemlaPipeline()
    result = pipe.fit_evaluate(X)

    print(result.summary())

    report_path = export_markdown_report(
        result,
        output_path="reports/gemla_transport_report.md",
    )

    print()
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
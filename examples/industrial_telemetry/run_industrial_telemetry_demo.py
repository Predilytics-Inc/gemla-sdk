from gemla.data import make_industrial_telemetry
from gemla.pipelines import GemlaPipeline
from gemla.reports import export_markdown_report


def main() -> None:
    X = make_industrial_telemetry()

    pipe = GemlaPipeline()
    result = pipe.fit_evaluate(X)

    print("GEMLA Industrial Telemetry Demo")
    print("-------------------------------")
    print(result.summary())

    report_path = export_markdown_report(
        result,
        output_path="reports/industrial_telemetry_report.md",
        title="GEMLA Industrial Telemetry Report",
    )

    print()
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
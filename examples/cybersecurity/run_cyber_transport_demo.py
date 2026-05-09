from gemla.data import make_cyber_event_transport
from gemla.pipelines import GemlaPipeline
from gemla.reports import export_markdown_report


def main() -> None:
    X = make_cyber_event_transport()

    pipe = GemlaPipeline()
    result = pipe.fit_evaluate(X)

    print("GEMLA Cyber Event Transport Demo")
    print("--------------------------------")
    print(result.summary())

    report_path = export_markdown_report(
        result,
        output_path="reports/cyber_event_transport_report.md",
        title="GEMLA Cyber Event Transport Report",
    )

    print()
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
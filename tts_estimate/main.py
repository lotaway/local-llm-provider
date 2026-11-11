import argparse
import json
import logging
from typing import Any, Dict, List

from .config.settings import AppConfig
from .infrastructure.evaluator_factory import build_quality_evaluator
from .application.evaluator_service import TTSQualityEvaluator
from .report_summary import build_report, summarize_reports

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def bootstrap() -> TTSQualityEvaluator:
    return build_quality_evaluator(AppConfig())


def run_evaluation(
    evaluator: TTSQualityEvaluator,
    ref_text: str,
    ref_audio: str,
    gen_audio: str,
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    result = evaluator.evaluate(ref_text, ref_audio, gen_audio, thresholds)

    return build_report(result)


def build_cases(args: argparse.Namespace) -> List[Dict[str, str]]:
    if args.cases:
        with open(args.cases, "r", encoding="utf-8") as handle:
            cases = json.load(handle)
        if not isinstance(cases, list) or not cases:
            raise ValueError("cases file must contain a non-empty JSON array")
        return cases

    return [
        {
            "ref_text": args.ref_text or "Hello world",
            "ref_audio": args.ref_audio or "path/to/ref.wav",
            "gen_audio": args.gen_audio or "path/to/gen.wav",
        }
    ]


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TTS quality")
    parser.add_argument("--cases", default=None, help="JSON file with evaluation cases")
    parser.add_argument("--ref-text", default=None)
    parser.add_argument("--ref-audio", default=None)
    parser.add_argument("--gen-audio", default=None)
    parser.add_argument("--wer-max", type=float, default=None)
    parser.add_argument("--dnsmos-min", type=float, default=None)
    parser.add_argument("--similarity-min", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: List[str] = None) -> None:
    args = parse_args(argv)
    evaluator = bootstrap()
    thresholds = {
        "wer_max": args.wer_max if args.wer_max is not None else AppConfig().thresholds.wer_max,
        "dnsmos_min": args.dnsmos_min if args.dnsmos_min is not None else AppConfig().thresholds.dnsmos_min,
        "similarity_min": args.similarity_min if args.similarity_min is not None else AppConfig().thresholds.similarity_min,
    }
    reports = []
    for case in build_cases(args):
        reports.append(
            run_evaluation(evaluator, case["ref_text"], case["ref_audio"], case["gen_audio"], thresholds)
        )
    logger.info(
        "tts_evaluation_batch_completed",
        extra={"summary": summarize_reports(reports)},
    )

if __name__ == "__main__":
    main()

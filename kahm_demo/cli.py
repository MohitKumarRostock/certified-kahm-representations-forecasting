"""Command-line entry points for forecasting and verified fitted-risk assessment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ._util import read_json, write_json
from .contract import validate_contract
from .datasets import generate_bundle, materialize_calibration, verify_bundle
from .oracle import exact_risk, parse_prediction_table, reference_summary


def _emit(value, output=None):
    if output:
        path = Path(output)
        if path.exists():
            raise FileExistsError(f"output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, value)
    else:
        print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="KAHM demonstrator: forecasting, frozen models, and verified risk certificates")
    commands = parser.add_subparsers(dest="command", required=True)
    ref = commands.add_parser("reference", help="show analytically checked benchmark quantities")
    ref.add_argument("--output", type=Path)
    generate = commands.add_parser("generate", help="create development data and reserve calibration")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--master-seed", type=int, default=20261110)
    generate.add_argument("--run-id", default="primary-v0.1")
    generate.add_argument("--training-count", type=int, default=4096)
    generate.add_argument("--validation-count", type=int, default=1024)
    verify = commands.add_parser("verify", help="verify bundle identity and structure")
    verify.add_argument("--bundle", type=Path, required=True)
    materialize = commands.add_parser("materialize-calibration", help="materialize the reserved stream after model freeze")
    materialize.add_argument("--bundle", type=Path, required=True)
    materialize.add_argument("--freeze", type=Path, required=True)
    audit = commands.add_parser("audit-predictions", help="compute exact population reference risk from a supplied prediction table")
    audit.add_argument("--predictions", type=Path, required=True)
    audit.add_argument("--output", type=Path)
    encoders = commands.add_parser("fit-encoders", help="fit training-only Gaussian KAHM development banks")
    encoders.add_argument("--bundle", type=Path, required=True)
    encoders.add_argument("--output", type=Path, required=True)
    encoders.add_argument("--caps", type=int, nargs="+", default=[1, 2])
    verify_encoders = commands.add_parser("verify-encoders", help="check saved encoders and replay all supported feature rows")
    verify_encoders.add_argument("--encoders", type=Path, required=True)
    verify_encoders.add_argument("--bundle", type=Path)
    encode = commands.add_parser("encode", help="inspect a history's raw, MEM, and LOCAL features")
    encode.add_argument("--encoder", type=Path, required=True)
    encode.add_argument("--history", type=float, nargs="+", required=True)
    encode.add_argument("--output", type=Path)
    fit_forecasters = commands.add_parser("fit-forecasters", help="fit and validation-select the matched RAW/MEM/LOCAL forecasting family")
    fit_forecasters.add_argument("--bundle", type=Path, required=True)
    fit_forecasters.add_argument("--encoders", type=Path, required=True)
    fit_forecasters.add_argument("--output", type=Path, required=True)
    verify_forecasters = commands.add_parser("verify-forecasters", help="recompute development losses/selection and replay the selected models")
    verify_forecasters.add_argument("--models", type=Path, required=True)
    verify_forecasters.add_argument("--bundle", type=Path, required=True)
    verify_forecasters.add_argument("--encoders", type=Path, required=True)
    forecast = commands.add_parser("forecast", help="predict the next observation from one observed history")
    forecast.add_argument("--model", type=Path, required=True)
    forecast.add_argument("--history", type=float, nargs="+", required=True)
    forecast.add_argument("--output", type=Path)
    freeze = commands.add_parser("freeze-forecasters", help="freeze the selected predictors and calibration protocol without generating calibration")
    freeze.add_argument("--bundle", type=Path, required=True)
    freeze.add_argument("--encoders", type=Path, required=True)
    freeze.add_argument("--models", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    verify_freeze = commands.add_parser("verify-freeze", help="verify the frozen models, complete development snapshot, and calibration protocol")
    verify_freeze.add_argument("--freeze", type=Path, required=True)
    verify_freeze.add_argument("--bundle", type=Path)
    forecast_frozen = commands.add_parser("forecast-frozen", help="forecast with an identity-checked member of the frozen family")
    forecast_frozen.add_argument("--freeze", type=Path, required=True)
    forecast_frozen.add_argument("--candidate", choices=("raw", "kahm_mem", "kahm_local"), required=True)
    forecast_frozen.add_argument("--history", type=float, nargs="+", required=True)
    forecast_frozen.add_argument("--bundle", type=Path)
    forecast_frozen.add_argument("--output", type=Path)
    certify = commands.add_parser("certify", help="assess the frozen family using its reserved or archived calibration data")
    certify.add_argument("--bundle", type=Path, required=True)
    certify.add_argument("--freeze", type=Path, required=True)
    certify.add_argument("--output", type=Path, required=True)
    verify_certificate = commands.add_parser("verify-certificate", help="recompute exact calibration losses and directly verify saved upper endpoints")
    verify_certificate.add_argument("--certificate", type=Path, required=True)
    verify_certificate.add_argument("--bundle", type=Path, required=True)
    verify_certificate.add_argument("--freeze", type=Path, required=True)
    audit_certificate = commands.add_parser("audit-certificate", help="audit all frozen predictors against the exact benchmark after certificate verification")
    audit_certificate.add_argument("--certificate", type=Path, required=True)
    audit_certificate.add_argument("--bundle", type=Path, required=True)
    audit_certificate.add_argument("--freeze", type=Path, required=True)
    audit_certificate.add_argument("--output", type=Path, required=True)
    verify_audit = commands.add_parser("verify-audit", help="recompute exact population risks and verify all saved certificate comparisons")
    verify_audit.add_argument("--audit", type=Path, required=True)
    verify_audit.add_argument("--certificate", type=Path, required=True)
    verify_audit.add_argument("--bundle", type=Path, required=True)
    verify_audit.add_argument("--freeze", type=Path, required=True)
    tutorial = commands.add_parser("tutorial", help="inspect archived primary evidence, or explicitly re-verify and replay it")
    tutorial.add_argument("--project", type=Path, default=Path("."))
    tutorial.add_argument("--mode", choices=("inspect", "verify"), default="inspect")
    tutorial.add_argument("--history", type=float, nargs=2, default=[0.25, 0.75])
    reproduce = commands.add_parser("reproduce", help="run one fresh development, freeze, certificate, and audit experiment")
    reproduce.add_argument("--output", type=Path, required=True)
    reproduce.add_argument("--master-seed", type=int, default=20261110)
    reproduce.add_argument("--run-id", default="tutorial-v0.1")
    reproduce.add_argument("--training-count", type=int, default=4096)
    reproduce.add_argument("--validation-count", type=int, default=1024)
    args = parser.parse_args(argv)
    try:
        validate_contract()
        if args.command == "reference":
            _emit(reference_summary(), args.output)
        elif args.command == "generate":
            _emit(generate_bundle(args.output, master_seed=args.master_seed, run_id=args.run_id,
                                  training_count=args.training_count, validation_count=args.validation_count))
        elif args.command == "verify":
            _emit(verify_bundle(args.bundle))
        elif args.command == "materialize-calibration":
            _emit(materialize_calibration(args.bundle, args.freeze))
        elif args.command == "audit-predictions":
            table = read_json(args.predictions)
            risk = exact_risk(parse_prediction_table(table))
            _emit({"schema": "kahm.population_reference_audit.v1",
                   "candidate_id": table["candidate_id"],
                   "risk": {"numerator": str(risk.numerator), "denominator": str(risk.denominator),
                            "decimal_approximation": float(risk)},
                   "interpretation": "Exact finite-support reference calculation. For the primary family, use audit-certificate after the existing calibration assessment has been issued and verified."}, args.output)
        elif args.command == "fit-encoders":
            from .encoder_workflow import fit_development_encoders
            _emit(fit_development_encoders(args.bundle, args.output, caps=tuple(args.caps)))
        elif args.command == "verify-encoders":
            from .encoder_workflow import verify_encoder_bundle
            _emit(verify_encoder_bundle(args.encoders, dataset_bundle=args.bundle))
        elif args.command == "encode":
            from .encoder_state import load_encoder
            encoder = load_encoder(args.encoder)
            row = encoder.encode_one(args.history)
            _emit({"history": args.history, "feature_widths": encoder.feature_widths,
                   "raw": row.raw.tolist(), "kahm_mem": row.kahm_mem.tolist(),
                   "kahm_local": row.kahm_local.tolist(), "scores": row.scores.tolist(),
                   "log_normalizers": row.log_normalizers.tolist(),
                   "reconstructions": row.reconstructions.tolist(),
                   "model_diagnostics": list(row.model_diagnostics),
                   "interpretation": "Encoder features, before decoder scaling; no forecast or performance certificate."}, args.output)
        elif args.command == "fit-forecasters":
            from .forecast_workflow import fit_development_forecasters
            _emit(fit_development_forecasters(args.bundle, args.encoders, args.output))
        elif args.command == "verify-forecasters":
            from .forecast_workflow import verify_forecast_bundle
            _emit(verify_forecast_bundle(args.models, dataset_bundle=args.bundle, encoder_bundle=args.encoders))
        elif args.command == "forecast":
            from .forecast_state import load_forecaster
            model = load_forecaster(args.model)
            result = model.inspect_one(args.history)
            _emit({"candidate_id": model.candidate_id, "bank_id": model.bank_id,
                   "history": args.history, "prediction": result.prediction,
                   "prediction_hex": result.prediction.hex(),
                   "unclipped_prediction": result.unclipped_prediction,
                   "clipped": result.clipped, "encoder_fallbacks": list(result.encoder_fallbacks),
                   "status": read_json(args.model)["status"],
                   "interpretation": "One-step forecast. Use verify-certificate to inspect a bound for the corresponding frozen family."}, args.output)
        elif args.command == "freeze-forecasters":
            from .freeze import freeze_forecasters
            _emit(freeze_forecasters(args.bundle, args.encoders, args.models, args.output))
        elif args.command == "verify-freeze":
            from .freeze import verify_frozen_bundle
            result = verify_frozen_bundle(args.freeze, dataset_bundle=args.bundle)
            # The model-freeze verifier does not inspect calibration outcomes.
            # Its historical field must not imply that no separate assessment
            # exists now. This command verifies the frozen model interface only.
            result.pop("statistical_certificate_issued", None)
            result["risk_certificate_checked"] = False
            result["interpretation"] = "The frozen computational family and protocol pass verification. Use verify-certificate to inspect the separate risk assessment."
            _emit(result)
        elif args.command == "forecast-frozen":
            from .freeze import load_frozen_forecaster
            frozen = load_frozen_forecaster(args.freeze, args.candidate, dataset_bundle=args.bundle)
            result = frozen.inspect_one(args.history)
            _emit({"candidate_id": frozen.candidate_id, "bank_id": frozen.bank_id,
                   "history": args.history, "prediction": result.prediction,
                   "prediction_hex": result.prediction.hex(),
                   "unclipped_prediction": result.unclipped_prediction,
                   "clipped": result.clipped, "encoder_fallbacks": list(result.encoder_fallbacks),
                   "freeze_record_sha256": frozen.freeze_record_sha256,
                   "calibration_protocol_sha256": frozen.calibration_protocol_sha256,
                   "status": "frozen", "risk_certificate_checked": False,
                   "interpretation": "Prediction by a verified frozen model. This command does not inspect a risk certificate; use verify-certificate for its assessment."}, args.output)
        elif args.command == "certify":
            from .certificate_workflow import certify_frozen_family
            _emit(certify_frozen_family(args.bundle, args.freeze, args.output))
        elif args.command == "verify-certificate":
            from .certificate_workflow import verify_certificate_bundle
            _emit(verify_certificate_bundle(args.certificate, dataset_bundle=args.bundle, freeze_path=args.freeze))
        elif args.command == "audit-certificate":
            from .audit_workflow import audit_certificate_family
            _emit(audit_certificate_family(args.certificate, dataset_bundle=args.bundle, freeze_path=args.freeze, output=args.output))
        elif args.command == "verify-audit":
            from .audit_workflow import verify_audit_bundle
            _emit(verify_audit_bundle(args.audit, certificate_path=args.certificate, dataset_bundle=args.bundle, freeze_path=args.freeze))
        elif args.command == "tutorial":
            from .tutorial import history_from_observations, inspect_primary, verify_primary
            history = history_from_observations(args.history)
            action = inspect_primary if args.mode == "inspect" else verify_primary
            _emit(action(args.project, history=history))
        elif args.command == "reproduce":
            from .reproduce import reproduce_demo
            result = reproduce_demo(
                args.output, master_seed=args.master_seed, run_id=args.run_id,
                training_count=args.training_count, validation_count=args.validation_count,
                progress=lambda stage: print(f"[reproduce] {stage}", file=sys.stderr, flush=True),
            )
            _emit(result)
        return 0
    except (ValueError, TypeError, KeyError, OSError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")


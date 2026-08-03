# Mac release checklist

- [ ] Clean source tree contains no `.auth.json` or test projects.
- [ ] `requirements.txt`, `code/scripts/requirements.txt`, `pyproject.toml`, README and CI all target Python 3.12.
- [ ] `python -m code.scripts.smoke_test` passes.
- [ ] Source launcher opens PAMalytics from a clean `PAMALYTICS_HOME`.
- [ ] Unsigned `PAMalytics.app` builds locally.
- [ ] Unsigned app opens on the build Mac.
- [ ] Signed app passes `codesign --verify --deep --strict --verbose=2`.
- [ ] DMG is notarised.
- [ ] DMG is stapled.
- [ ] Gatekeeper assessment passes.
- [ ] Clean Mac test passes from Downloads.
- [ ] Clean Mac test passes from Applications.
- [ ] Release notes include version, supported macOS, known issues and citation guidance.

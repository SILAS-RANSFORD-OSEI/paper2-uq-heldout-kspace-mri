# Artifact and Data Policy

## Files permitted in Git

- source code;
- YAML experiment configurations;
- public manifests without protected paths;
- unit tests;
- protocol and decision documents;
- small CSV/JSON summaries;
- figure-generation scripts;
- checksums and provenance records.

## Files prohibited from Git

- fastMRI HDF5 data;
- patient-identifying information;
- private Google Drive paths;
- large NPZ reliability caches;
- unrestricted model checkpoints;
- access tokens or credentials;
- raw experiment logs containing sensitive paths.

## Large artifact storage

Large caches, predictions, and checkpoints remain on Google Drive during
development. Publicly releasable models and result archives may later be
deposited in a versioned release or persistent repository.

## Required provenance

Every scientific result must record:

- Git commit;
- resolved configuration;
- input manifest hash;
- software environment;
- random seed;
- runtime device;
- output checksums;
- PASS, WARN, or FAIL status.

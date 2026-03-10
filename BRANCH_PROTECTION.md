# Branch Protection Rules

Configure these settings manually in GitHub repository settings for the `main` branch.

## Required Status Checks

Require all of the following checks to pass before merge:

- `lint-and-typecheck`
- `test-django`
- `test-ml-services`
- `validate-helm`

## Pull Request Requirements

- Require at least 1 approving review
- Dismiss stale pull request approvals when new commits are pushed

## Safety Controls

- Disallow force pushes
- Do not allow bypassing branch protections (including administrators)
- Automatically delete head branches after merge

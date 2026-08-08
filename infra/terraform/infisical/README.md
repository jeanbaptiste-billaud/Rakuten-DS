# Infisical Terraform Bootstrap

This Terraform stack owns Infisical configuration for Rakuten DS V2:

- project
- secret folders
- machine identities
- one Universal Auth method for the Terraform deployment identity
- identity project memberships
- path-scoped identity privileges

Secret values are not managed here. They remain seeded by Ansible from
Ansible Vault to avoid storing application secrets in Terraform configuration.

Application identities are created without Universal Auth credentials. Their
authentication is intended to be provided by SPIRE. Only `deploy-id` keeps a
Universal Auth client ID and client secret so Terraform can access Infisical.

The Terraform state, generated output, and runtime environment files still
contain that Terraform credential. Treat them as sensitive local material and
keep them out of Git until they are encrypted through the Ansible-managed
local protection flow.

The stack outputs the public `identity_ids` map. Pass it to the Docker stack as
`infisical_identity_ids`; no workload client secret is exported.

```bash
terraform -chdir=infra/terraform/infisical output -json identity_ids > identity-ids.json
```

Infisical provider 0.16 does not expose a SPIFFE Auth resource. Attach SPIFFE
Auth to each identity through Infisical API or UI with:

- trust domain: `rakuten.local`;
- allowed audience: `infisical`;
- allowed ID: `spiffe://rakuten.local/workload/<workload>`;
- a short access-token TTL appropriate for one worker execution.
- trust bundle profile: `HTTPS Web Bundle`;
- bundle endpoint URL: `https://spire-server:8443`;
- root CA certificate: `~/.config/rakuten/spire/tls/ca.crt`;
- bundle refresh hint: `3600` seconds.

Existing local state and old `terraform-output.json` files may still contain
Universal Auth secrets. Rotate those credentials, remove the old artifacts
securely, and move the state to a protected backend. Never commit them.

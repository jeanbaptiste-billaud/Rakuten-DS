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

# Infisical Terraform Bootstrap

This Terraform stack owns Infisical configuration for Rakuten DS V2:

- project
- secret folders
- machine identities
- Universal Auth methods
- identity project memberships
- path-scoped identity privileges

Secret values are not managed here. They remain seeded by Ansible from
Ansible Vault to avoid storing application secrets in Terraform configuration.

The Terraform state still contains Universal Auth client secrets. Treat the
state as sensitive material. Keep the local state and generated tfvars out of
Git.

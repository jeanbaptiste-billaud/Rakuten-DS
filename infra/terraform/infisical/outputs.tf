output "project_id" {
  value = infisical_project.rakuten.id
}

output "project_slug" {
  value = infisical_project.rakuten.slug
}

output "terraform_identity_universal_auth" {
  sensitive = true
  value = {
    identity_name = var.terraform_identity_name
    client_id     = infisical_identity_universal_auth_client_secret.client_secrets[var.terraform_identity_name].client_id
    client_secret = infisical_identity_universal_auth_client_secret.client_secrets[var.terraform_identity_name].client_secret
  }
}

output "identity_ids" {
  description = "Public identity UUIDs to inject into SPIFFE-authenticated workloads."
  value = {
    for name, identity in infisical_identity.identities : name => identity.id
  }
}


output "project_id" {
  value = infisical_project.rakuten.id
}

output "project_slug" {
  value = infisical_project.rakuten.slug
}

output "identity_universal_auth" {
  sensitive = true
  value = {
    for name, secret in infisical_identity_universal_auth_client_secret.client_secrets : name => {
      client_id     = secret.client_id
      client_secret = secret.client_secret
      path          = var.identities[name].path
    }
  }
}


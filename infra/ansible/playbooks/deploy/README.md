# Deploy Recovery Playbooks

## Infisical

Use this playbook to recover the Infisical service stack after a host restart,
power loss, or any other interruption that left the containers stopped:

```bash
ansible-playbook infra/ansible/playbooks/deploy/recover_infisical.yaml --limit deploy
```

The playbook reuses the bootstrap tasks that ensure:

- the Infisical PostgreSQL dependency exists
- the Infisical Redis dependency exists
- the Infisical container is started
- the Infisical Terraform configuration is reapplied when the local bootstrap
  state is present


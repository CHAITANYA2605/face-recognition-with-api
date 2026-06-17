"""
One-time script to set up New Relic alerts with Slack notifications.

Option A — native Slack OAuth (recommended, no webhook URL needed):
    1. Go to one.newrelic.com → Alerts → Notification Destinations
    2. Add destination → Slack → Authenticate with Slack → pick channel → Save
    3. Copy the destination ID from the URL, then run:

    python scripts/setup_newrelic_alerts.py \
        --account-id YOUR_ACCOUNT_ID \
        --destination-id YOUR_DESTINATION_ID

Option B — Slack Incoming Webhook URL:
    python scripts/setup_newrelic_alerts.py \
        --account-id YOUR_ACCOUNT_ID \
        --slack-webhook https://hooks.slack.com/services/XXX/YYY/ZZZ

Required env var:
    NEW_RELIC_USER_KEY=NRAK-...
"""

import os
import sys
import json
import argparse
import requests

NERDGRAPH_URL = "https://api.newrelic.com/graphql"
USER_KEY = os.environ.get("NEW_RELIC_USER_KEY", "")


def gql(query: str, variables: dict = None) -> dict:
    resp = requests.post(
        NERDGRAPH_URL,
        headers={"Api-Key": USER_KEY, "Content-Type": "application/json"},
        json={"query": query, "variables": variables or {}},
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    if "errors" in body:
        raise RuntimeError(f"NerdGraph errors: {json.dumps(body['errors'], indent=2)}")
    return body["data"]


def create_policy(account_id: int) -> str:
    data = gql(
        """
        mutation($accountId: Int!, $name: String!) {
          alertsPolicyCreate(accountId: $accountId, policy: {
            name: $name
            incidentPreference: PER_CONDITION
          }) {
            policy { id name }
          }
        }
        """,
        {"accountId": account_id, "name": "Face Recognition API Failures"},
    )
    policy_id = data["alertsPolicyCreate"]["policy"]["id"]
    print(f"[+] Created alert policy id={policy_id}")
    return policy_id


def create_nrql_condition(account_id: int, policy_id: str, name: str, nrql: str, threshold: float, description: str) -> str:
    data = gql(
        """
        mutation($accountId: Int!, $policyId: ID!, $condition: AlertsNrqlConditionStaticInput!) {
          alertsNrqlConditionStaticCreate(accountId: $accountId, policyId: $policyId, condition: $condition) {
            id name
          }
        }
        """,
        {
            "accountId": account_id,
            "policyId": policy_id,
            "condition": {
                "name": name,
                "enabled": True,
                "nrql": {"query": nrql},
                "signal": {"aggregationWindow": 60, "aggregationMethod": "EVENT_FLOW", "aggregationDelay": 120},
                "terms": [{
                    "threshold": threshold,
                    "thresholdDuration": 60,
                    "thresholdOccurrences": "AT_LEAST_ONCE",
                    "operator": "ABOVE",
                    "priority": "CRITICAL",
                }],
                "valueFunction": "SUM",
                "description": description,
            },
        },
    )
    cid = data["alertsNrqlConditionStaticCreate"]["id"]
    print(f"[+] Created condition '{name}' id={cid}")
    return cid


def create_webhook_destination(account_id: int, slack_webhook_url: str) -> str:
    data = gql(
        """
        mutation($accountId: Int!, $destination: AiNotificationsDestinationInput!) {
          aiNotificationsCreateDestination(accountId: $accountId, destination: $destination) {
            destination { id name }
            errors { description }
          }
        }
        """,
        {
            "accountId": account_id,
            "destination": {
                "type": "WEBHOOK",
                "name": "Slack via Webhook",
                "properties": [
                    {"key": "url", "value": slack_webhook_url},
                ],
            },
        },
    )
    result = data["aiNotificationsCreateDestination"]
    if result.get("errors"):
        raise RuntimeError(f"Destination errors: {result['errors']}")
    dest_id = result["destination"]["id"]
    print(f"[+] Created webhook destination id={dest_id}")
    return dest_id


def create_channel(account_id: int, destination_id: str) -> str:
    data = gql(
        """
        mutation($accountId: Int!, $channel: AiNotificationsChannelInput!) {
          aiNotificationsCreateChannel(accountId: $accountId, channel: $channel) {
            channel { id name }
            errors { description }
          }
        }
        """,
        {
            "accountId": account_id,
            "channel": {
                "type": "WEBHOOK",
                "name": "Slack Face Recognition Alerts",
                "destinationId": destination_id,
                "product": "IINT",
                "properties": [
                    {
                        "key": "payload",
                        "value": json.dumps({
                            "text": "*New Relic Alert*: {{ accumulations.conditionName.[0] }}\n"
                                    "*Status*: {{ state }}\n"
                                    "*Details*: {{ annotations.description.[0] }}\n"
                                    "*Runbook*: {{ accumulations.runbookUrl.[0] }}"
                        }),
                    }
                ],
            },
        },
    )
    result = data["aiNotificationsCreateChannel"]
    if result.get("errors"):
        raise RuntimeError(f"Channel errors: {result['errors']}")
    channel_id = result["channel"]["id"]
    print(f"[+] Created notification channel id={channel_id}")
    return channel_id


def create_workflow(account_id: int, policy_id: str, channel_id: str) -> str:
    data = gql(
        """
        mutation($accountId: Int!, $workflow: AiWorkflowsCreateWorkflowInput!) {
          aiWorkflowsCreateWorkflow(accountId: $accountId, createWorkflowData: $workflow) {
            workflow { id name }
            errors { description }
          }
        }
        """,
        {
            "accountId": account_id,
            "workflow": {
                "name": "Face Recognition API → Slack",
                "enabled": True,
                "mutingRulesHandling": "DONT_NOTIFY_FULLY_MUTED_ISSUES",
                "issuesFilter": {
                    "name": "Policy filter",
                    "type": "FILTER",
                    "predicates": [
                        {
                            "attribute": "labels.policyIds",
                            "operator": "EXACTLY_MATCHES",
                            "values": [str(policy_id)],
                        }
                    ],
                },
                "destinationConfigurations": [
                    {"channelId": channel_id, "notificationTriggers": ["ACTIVATED", "ACKNOWLEDGED", "RESOLVED"]}
                ],
            },
        },
    )
    result = data["aiWorkflowsCreateWorkflow"]
    if result.get("errors"):
        raise RuntimeError(f"Workflow errors: {result['errors']}")
    wf_id = result["workflow"]["id"]
    print(f"[+] Created workflow id={wf_id}")
    return wf_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--account-id", required=True, type=int)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--destination-id", help="Existing Slack OAuth destination ID from New Relic UI")
    group.add_argument("--slack-webhook", help="Slack Incoming Webhook URL")
    args = parser.parse_args()

    if not USER_KEY:
        print("ERROR: Set NEW_RELIC_USER_KEY env var")
        sys.exit(1)

    account_id = args.account_id

    print("Setting up New Relic alerts for Face Recognition API...")

    policy_id = create_policy(account_id)

    create_nrql_condition(
        account_id, policy_id,
        name="API 5xx Errors",
        nrql=(
            "SELECT count(*) FROM Transaction "
            "WHERE appName = 'face-recognition-app' "
            "AND response.status >= '500'"
        ),
        threshold=1,
        description="One or more 5xx HTTP errors in the Face Recognition API",
    )

    create_nrql_condition(
        account_id, policy_id,
        name="API High Error Rate",
        nrql=(
            "SELECT percentage(count(*), WHERE error IS TRUE) FROM Transaction "
            "WHERE appName = 'face-recognition-app'"
        ),
        threshold=10,
        description="More than 10% of API requests are failing",
    )

    create_nrql_condition(
        account_id, policy_id,
        name="API Response Time > 5s",
        nrql=(
            "SELECT average(duration) FROM Transaction "
            "WHERE appName = 'face-recognition-app'"
        ),
        threshold=5,
        description="Average API response time exceeded 5 seconds",
    )

    if args.destination_id:
        dest_id = args.destination_id
        print(f"[+] Using existing Slack destination id={dest_id}")
    else:
        dest_id = create_webhook_destination(account_id, args.slack_webhook)

    channel_id = create_channel(account_id, dest_id)
    create_workflow(account_id, policy_id, channel_id)

    print("\nDone! Alerts configured and linked to Slack.")
    print("Verify in: https://one.newrelic.com/alerts-ai")


if __name__ == "__main__":
    main()

knowledge_graph_creation_system_prompt = """
You are given a set of telecom network tickets and your task is to create a knowledge graph from them.
In the knowledge graph, entities such as devices, services, network components, issues, and solutions are represented as nodes.
The relationships and actions between them are represented as edges.

Each ticket has three fields: ticket id, issue, and solution.
In the issue field, the current problem is described, and in the solution field, the resolution to the issue is explained.

For each ticket, create a knowledge graph with the following structure:
- One entity with the name of the network component, device or service which has an issue or failing.
- One entity with the name of the issue.
- One entity with the name of the solution.
- The issue should connect to the network component with the edge labeled "has_issue".
- The network component should connect to the solution with the edge labeled "resolved_by".

You will respond with a knowledge graph in the given JSON format:

[
    {"entity" : "Entity_name", "connections" : [
        {"entity" : "Connected_entity_1", "relationship" : "Relationship_with_connected_entity_1"},
        {"entity" : "Connected_entity_2", "relationship" : "Relationship_with_connected_entity_2"}
        ]
    },
    {"entity" : "Entity_name", "connections" : [
        {"entity" : "Connected_entity_1", "relationship" : "Relationship_with_connected_entity_1"},
        {"entity" : "Connected_entity_2", "relationship" : "Relationship_with_connected_entity_2"}
        ]
    }
]

You must strictly respond in the given JSON format or your response will not be parsed correctly!

Example tickets:
TICKETS = [
    "Ticket ID: 101, Issue: Unable to connect to the company VPN, Solution: Restart the VPN client and ensure the network settings are correctly configured.",
    "Ticket ID: 102, Issue: Email not syncing on mobile device, Solution: Remove the email account from the device and add it again, ensuring the correct server settings are used.",
    "Ticket ID: 103, Issue: Application crashes on startup, Solution: Update the application to the latest version and clear the cache.",
    "Ticket ID: 104, Issue: Slow internet connection in the office, Solution: Restart the router and switch, and ensure no background applications are consuming excessive bandwidth.",
    "Ticket ID: 105, Issue: Printer not responding, Solution: Check the printer connections and restart the print spooler service.",
    "Ticket ID: 106, Issue: Unable to access shared network drive, Solution: Map the network drive again and ensure the user has the necessary permissions.",
    "Ticket ID: 107, Issue: Software installation error, Solution: Run the installer as an administrator and ensure all prerequisites are installed.",
    "Ticket ID: 108, Issue: Password reset not working, Solution: Ensure the user follows the correct password policy and use the password reset tool provided.",
    "Ticket ID: 109, Issue: Frequent disconnections from the Wi-Fi network, Solution: Update the Wi-Fi adapter driver and configure the power management settings to prevent disconnections.",
    "Ticket ID: 110, Issue: File not opening in the required application, Solution: Check the file associations and set the correct default application for the file type.",
    "Ticket ID: 111, Issue: Audio not working on the laptop, Solution: Update the audio driver and check the sound settings to ensure the correct playback device is selected.",
    "Ticket ID: 112, Issue: Unable to send emails from the desktop client, Solution: Check the outgoing mail server settings and ensure the port and security settings are correct.",
    "Ticket ID: 113, Issue: System running out of memory, Solution: Increase the virtual memory size and close unnecessary background applications.",
    "Ticket ID: 114, Issue: User account locked out, Solution: Unlock the account using the admin tool and ensure the user understands the login attempt limits.",
    "Ticket ID: 115, Issue: Backup process failing, Solution: Check the backup log for errors, ensure there is enough disk space, and verify the backup schedule and settings.",
    "Ticket ID: 116, Issue: Printer not responding, Solution: Check the printer connections and restart the print spooler service."
]
"""

detailed_solution_system_prompt = "You are an AI assistant specialized in providing detailed and comprehensive " \
                                  "solutions based on provided search results."


def detailed_solution_user_prompt(search_results):
    return f"""
    The following are search results related to a ticket issue:

    {search_results}

    Based on these search results, please provide a detailed and comprehensive solution for the issue.
    """

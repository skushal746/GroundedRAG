"""Generate the binary sample documents (PDF product guide, DOCX billing policy).

The Markdown FAQ, HTML API guide and TXT release notes are checked in as plain files.
Run: `uv run python scripts/make_sample_docs.py`
"""

from pathlib import Path

import docx
import pymupdf
from docx.shared import Pt

OUT = Path(__file__).resolve().parent.parent / "sample_docs"

CSS = """
body { font-family: sans-serif; font-size: 10.5pt; line-height: 1.35; }
h1 { font-size: 20pt; }
h2 { font-size: 15pt; }
h3 { font-size: 12.5pt; }
table { border-collapse: collapse; }
th, td { border: 1px solid #444; padding: 3px 5px; font-size: 9.5pt; }
"""

# One HTML string per page, so expected page numbers in the eval set are stable.
GUIDE_PAGES = [
    """
<h1>Orbitask Product Guide</h1>
<p>Version 4.3. This guide explains how to set up and use Orbitask, the task and project
management platform for growing teams.</p>
<h2>Getting started</h2>
<p>When you sign up you create a workspace. A workspace contains all of your team's
projects, members and settings, and it is billed as a single subscription.</p>
<p>To invite people, go to Settings &gt; Members &gt; Invite and enter their email addresses.
Invitations expire after 7 days. You can also share an invite link that admins can disable
at any time.</p>
<h3>Roles</h3>
<p>Every person in a workspace has one role. The Owner controls billing and can delete the
workspace; there is exactly one Owner. Admins manage members, integrations and
workspace settings. Members create and edit projects and tasks. Guests only see projects
shared with them.</p>
""",
    """
<h2>Projects and tasks</h2>
<p>Projects group related tasks. Each task has a title, description, assignee, due date,
priority (Low, Medium, High or Urgent), labels and custom fields. Tasks can have subtasks
nested up to 3 levels deep.</p>
<h3>Views</h3>
<p>Every project can be displayed as a List, Board (Kanban), Calendar or Gantt view. Gantt
view is available on the Team, Business and Enterprise plans. Saved view filters are
shared with everyone in the project.</p>
<h3>Dependencies and recurring tasks</h3>
<p>A task can be marked as blocked by another task. When the blocking task's due date
moves, Orbitask can shift dependent tasks automatically if "auto-schedule" is enabled on the
project. Recurring tasks are created again automatically when the previous occurrence is
completed, using daily, weekly, monthly or custom intervals.</p>
""",
    """
<h2>Automations</h2>
<p>Automations run actions when something happens in a project, for example "when a task
moves to Done, notify the project channel in Slack" or "when a task is overdue, raise its
priority to Urgent".</p>
<p>Each action an automation performs counts as one automation run. An automation with a
trigger and three actions therefore uses three runs each time it fires. The monthly
allowance for each plan is listed in the Plan limits table.</p>
<p>Automation runs reset on the 1st day of each calendar month at 00:00 UTC. Unused runs do
not roll over to the next month. When a workspace reaches its allowance, automations pause
until the next reset and Admins receive an email; no tasks are lost.</p>
""",
    """
<h2>Integrations and apps</h2>
<p>Orbitask integrates with Slack, Microsoft Teams, Google Drive, GitHub, GitLab and
Zapier. Admins connect integrations under Settings &gt; Integrations. The GitHub and GitLab
integrations link commits and pull requests to tasks when the task ID (for example
ORB-123) appears in the commit message.</p>
<h3>Mobile and desktop apps</h3>
<p>The mobile apps require iOS 16 or later or Android 10 or later. Desktop apps are
available for macOS 13 or later and Windows 10 or later. Push notifications can be
configured per project from the app's notification settings.</p>
""",
    """
<h2>Plan limits</h2>
<p>The table below lists the usage limits for each plan. Limits apply per workspace unless
stated otherwise.</p>
<table>
<tr><th>Limit</th><th>Free</th><th>Team</th><th>Business</th><th>Enterprise</th></tr>
<tr><td>Members</td><td>3</td><td>Unlimited</td><td>Unlimited</td><td>Unlimited</td></tr>
<tr><td>Projects</td><td>5</td><td>Unlimited</td><td>Unlimited</td><td>Unlimited</td></tr>
<tr><td>Storage</td><td>1 GB</td><td>100 GB</td><td>1 TB</td><td>Unlimited</td></tr>
<tr><td>Max file upload</td><td>25 MB</td><td>250 MB</td><td>250 MB</td><td>5 GB</td></tr>
<tr><td>Automation runs per month</td><td>100</td><td>5,000</td><td>25,000</td><td>250,000</td></tr>
<tr><td>Guest seats per paid seat</td><td>0</td><td>5</td><td>10</td><td>Unlimited</td></tr>
<tr><td>Version history</td><td>7 days</td><td>90 days</td><td>1 year</td><td>Unlimited</td></tr>
</table>
<p>If a workspace goes over its storage limit, existing files remain available but new
uploads are blocked until files are deleted or the plan is upgraded.</p>
""",
    """
<h2>Security and administration</h2>
<h3>Authentication controls</h3>
<p>Any user can enable two-factor authentication. On Business and Enterprise plans, Admins
can require two-factor authentication for every member. SAML single sign-on (SSO) with
Okta, Microsoft Entra ID or Google Workspace is available on Business and Enterprise.
Automated user provisioning with SCIM is available on Enterprise only.</p>
<h3>Audit log</h3>
<p>The audit log records sign-ins, permission changes, exports and deletions. It is
available on Business (retained for 1 year) and Enterprise (retained for 3 years).</p>
<h3>Deleting a workspace</h3>
<p>Only the Owner can delete a workspace, from Settings &gt; Workspace &gt; Delete. Deleted
workspaces can be restored by the Owner for 30 days. After 30 days all data is permanently
deleted and cannot be recovered.</p>
""",
]


def make_pdf(path: Path) -> None:
    """Render one HTML page per PDF page, then stamp a running footer on every page."""
    rect = pymupdf.paper_rect("a4")
    where = pymupdf.Rect(rect.x0 + 54, rect.y0 + 54, rect.x1 - 54, rect.y1 - 72)
    writer = pymupdf.DocumentWriter(str(path))
    for html in GUIDE_PAGES:
        story = pymupdf.Story(html=html, user_css=CSS)
        more = True
        while more:
            device = writer.begin_page(rect)
            more, _ = story.place(where)
            story.draw(device)
            writer.end_page()
    writer.close()

    doc = pymupdf.open(str(path))
    if doc.page_count != len(GUIDE_PAGES):
        raise RuntimeError(f"Guide overflowed: {doc.page_count} pages")
    for number, page in enumerate(doc, start=1):
        footer = f"Orbitask Product Guide - v4.3 - Page {number} of {doc.page_count}"
        page.insert_text((54, rect.height - 36), footer, fontsize=8, color=(0.4, 0.4, 0.4))
    doc.set_metadata({"title": "Orbitask Product Guide", "author": "Orbitask Inc."})
    doc.save(str(path) + ".tmp", garbage=3, deflate=True)
    doc.close()
    Path(str(path) + ".tmp").replace(path)


BILLING: list[tuple[str, str | list[list[str]]]] = [
    ("title", "Orbitask Billing & Refunds Policy"),
    ("p", "Effective date: 1 March 2026. This policy replaces all earlier versions."),
    ("h1", "Plans and pricing"),
    (
        "p",
        "Orbitask is billed per paid seat. You can pay monthly or annually; annual billing "
        "costs the equivalent of 10 months, so you get 2 months free.",
    ),
    (
        "table",
        [
            ["Plan", "Monthly billing", "Annual billing", "Notes"],
            ["Free", "$0", "$0", "Up to 3 members"],
            ["Team", "$12 per seat / month", "$10 per seat / month", "Minimum 3 seats"],
            ["Business", "$24 per seat / month", "$20 per seat / month", "Minimum 5 seats"],
            ["Enterprise", "Custom", "Custom", "Minimum 50 seats, annual contract only"],
        ],
    ),
    ("h1", "Billing cycles and payment"),
    ("h2", "Payment methods"),
    (
        "p",
        "We accept Visa, Mastercard, American Express and PayPal. Annual plans with a total "
        "value above $5,000 can also be paid by bank transfer (ACH or SEPA) against an invoice "
        "with 30-day payment terms.",
    ),
    ("h2", "Adding and removing seats"),
    (
        "p",
        "Seats added during a billing period are charged immediately, prorated for the "
        "remaining days of the period. Removed seats stay available until the end of the "
        "current period; no credit is issued for removed seats.",
    ),
    ("h2", "Upgrades and downgrades"),
    (
        "p",
        "Upgrades take effect immediately and you are charged the prorated difference. "
        "Downgrades take effect at the end of the current billing period, and features that "
        "are not included in the new plan stop working on that date.",
    ),
    ("h2", "Taxes"),
    (
        "p",
        "Prices exclude taxes. VAT is added for customers in the European Union unless a valid "
        "VAT ID is entered under Settings > Billing, in which case the reverse-charge "
        "mechanism applies.",
    ),
    ("h1", "Failed payments"),
    (
        "p",
        "If a payment fails we retry the card 3 times over 7 days and email the workspace "
        "Owner after each attempt. If the invoice is still unpaid 14 days after the due date, "
        "the workspace becomes read-only: members can view and export data but cannot create "
        "or edit tasks. After 60 days unpaid, the workspace is downgraded to the Free plan.",
    ),
    ("h1", "Refunds"),
    ("h2", "Annual plans"),
    (
        "p",
        "Annual subscriptions are eligible for a full refund if you request it within 30 days "
        "of the initial purchase or of an annual renewal. After 30 days annual subscriptions "
        "are non-refundable, but you can turn off auto-renewal at any time.",
    ),
    ("h2", "Monthly plans"),
    (
        "p",
        "Monthly subscriptions are not refundable, including for partially used months. The "
        "only exception is a charge made after you cancelled, which is refunded in full.",
    ),
    ("h2", "How to request a refund"),
    (
        "p",
        "The workspace Owner can request a refund under Settings > Billing > Request refund, or "
        "by emailing billing@orbitask.example from the Owner's email address. Approved refunds "
        "are returned to the original payment method within 5 to 10 business days.",
    ),
    ("h1", "Discounts"),
    (
        "p",
        "Registered nonprofit organisations receive 40% off the Team and Business plans when "
        "billed annually. Accredited schools and universities receive 50% off the Team and "
        "Business plans. Apply by emailing billing@orbitask.example with proof of status. "
        "Discounts cannot be combined with each other or with promotional offers.",
    ),
    ("h1", "Cancellation"),
    (
        "p",
        "When you cancel a paid plan, the workspace stays on that plan until the end of the "
        "billing period and then moves to the Free plan. Projects beyond the Free plan limit "
        "of 5 are archived (read-only) rather than deleted, and can be restored by upgrading.",
    ),
]


def make_docx(path: Path) -> None:
    document = docx.Document()
    document.styles["Normal"].font.size = Pt(11)
    document.core_properties.title = "Orbitask Billing & Refunds Policy"
    for kind, content in BILLING:
        if kind == "title":
            document.add_heading(str(content), level=0)
        elif kind == "h1":
            document.add_heading(str(content), level=1)
        elif kind == "h2":
            document.add_heading(str(content), level=2)
        elif kind == "p":
            document.add_paragraph(str(content))
        elif kind == "table" and isinstance(content, list):
            table = document.add_table(rows=len(content), cols=len(content[0]))
            table.style = "Table Grid"
            for r, row in enumerate(content):
                for c, value in enumerate(row):
                    table.cell(r, c).text = value
    document.save(str(path))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    make_pdf(OUT / "orbitask_product_guide.pdf")
    make_docx(OUT / "orbitask_billing_and_refunds.docx")
    print(f"Wrote sample documents to {OUT}")


if __name__ == "__main__":
    main()

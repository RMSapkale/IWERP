# V1 Smoke Test Prompt Set

## Summary / Q&A

1. `what is EPM?`
2. `what is GL?`
3. `what is RMCS?`
4. `what is payroll gratuity?`
Expected: grounded concept answer or clean refusal. Procedural substitution is a failure.
5. `what is three-way match in Oracle Fusion Procurement?`
Expected: grounded summary answer with citations.

## Procedure

1. `how to create custom ESS job?`
2. `How do you create supplier site setup in Oracle Fusion Payables?`

## Troubleshooting

1. `AP invoice validation failed due to account combination error. How to troubleshoot?`
2. `Cash Management bank statement import failed. Provide troubleshooting steps.`

## SQL

1. `Create an Oracle Fusion SQL query to extract AP invoice distribution details. Include Invoice Number, Supplier Name, Distribution Line Number, Distribution Amount, Natural Account Segment, Cost Center, Liability Account. Show only validated and accounted invoices.`
Expected: full requested field/filter coverage or safe refusal. Reduced-shape SQL is a failure.
2. `Generate Oracle Fusion SQL for posted GL journal lines including natural account and cost center segments.`

## Fast Formula

1. `Create a Fast Formula for sick leave accrual with DEFAULT handling, INPUTS, and RETURN logic.`
2. `Troubleshoot Fast Formula compile error: database item not found. Provide symptom, cause, and fix.`
3. `Create a Fast Formula for payroll gratuity eligibility and payout logic with explicit RETURN.`
Expected: grounded formula or clean refusal.

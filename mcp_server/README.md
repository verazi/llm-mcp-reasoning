AIO API
- Limitation (observed): When using aggregationLevel=month together with sentiment=true, a single request spanning more than ~4 months results in a 400 error from the API.
- Client behavior: The client automatically splits long ranges into chunks of up to 4 months each, issues multiple requests, and merges the results.
- Alignment: For monthly aggregation, the client automatically aligns startDate to the first day of the month and endDate to the last day of the month, to avoid boundary errors or inconsistent bucket calculations.
- Alternative approach: To avoid multiple requests, you can instead query with aggregationLevel=day and sentiment=true, then group the results by month on the client side. However, this produces larger single responses.
- year => 1 year per quire, month has to be the same
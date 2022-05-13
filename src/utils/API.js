export async function getAllData(params, url) {
    const _getAllData = async (params, startKey) => {
    if (startKey) {
        params.sortkey = startKey.SortKey.S
        params.partkey = startKey.PartitionKey.S
    }
    // assigning label to url
    url.search = new URLSearchParams(params).toString()

    return fetch(url).then(res => res.json())
    }
    let lastEvaluatedKey = null
    let rows = []
    do {
    const result = await _getAllData(params, lastEvaluatedKey)
    rows = rows.concat(result.Items)
    lastEvaluatedKey = result.LastEvaluatedKey
    } while (lastEvaluatedKey)
    return rows
}
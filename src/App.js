import React, {Component} from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';
import Link from '@mui/material/Link';
import Grid from '@mui/material/Grid';

import Container from '@mui/material/Container';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import Autocomplete from '@mui/material/Autocomplete';
import Pagination from '@mui/material/Pagination';
import CircularProgress from '@mui/material/CircularProgress';

const PREFIX = 'App';

function MediaCard(props) {

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography gutterBottom variant="h5" component="h2">
          <Link
            color="primary"
            target="_blank"
            rel="noopener noreferrer"
            href={props.article.url.S}
          >
            {props.article.title.S}
          </Link>
        </Typography>
        <Typography variant="body2" color="textSecondary" component="p">
          {props.article.abstract.S}
        </Typography>
        <Typography style={{paddingTop: "10px"}} variant="body2" color="textSecondary" component="p">
          Published in: {props.article.venue.S}
        </Typography>
      </CardContent>
    </Card>
  );
}

class Results extends Component {
  render() {
    return (
      <div></div>
    )
  }
}

class App extends Component {

  constructor(props) {
    super(props);
    this.onSelectionChange = this.onSelectionChange.bind(this);
  }

  onSelectionChange = (event, values) => {
    this.setState({
      selection: values,
      fetchInProgress: true
    }, () => {
      //if the selection is X'd out, just fetch original articles
      if (this.state.selection == null) {
        this.setState({ articlesToDisplay: [], fetchInProgress: false })
        return;
      }

      //querying the database by selected label
      const getAllData = async (params) => {
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

      const selection_label = this.state.selection.id
      const url = new URL('https://ardwrgr7s5.execute-api.us-east-2.amazonaws.com/v1/getarticles')
      const params = { level3: selection_label }

      getAllData(params).then((data) => {
        this.setState({ fetchInProgress: false });

        data = data.filter(object => {
          return parseFloat(object.score.S) > .3;
        });

        // sort papers by scores DESC.
        data.sort(function (a, b) {
          return parseFloat(b.score.S) - parseFloat(a.score.S);
        });

        this.setState({ articlesToDisplay: data })
      }).catch(console.log)
    });
  }

  render() {

    const articleCards = this.state.articlesToDisplay.map((article) =>
      <Grid item xs={12} key={article.SortKey.S}><MediaCard article={article} /></Grid>
    );

    return (
      <Container maxWidth="lg">
        <Box sx={{ my: 3 }}>
          <Grid
            container
            rowSpacing={1}
            justifyContent="space-between"
          >
          <Grid
            item
            order={{ sm: 1, md: 2 }}
          >
          <Box
            component="img"
            sx={{
              height: 100
            }}
            alt="PeTaL logo"
            src={process.env.PUBLIC_URL + '/petal-logo-text-white.png'}
          />
          </Grid>
          <Grid item>
            <Typography variant="h5" component="h1" gutterBottom>
              How does nature...
            </Typography>
            <Autocomplete
              id="function"
              options={this.state.functions.sort((a, b) => -b.level2.localeCompare(a.level2))}
              groupBy={(option) => option.level2}
              getOptionLabel={(option) => option.level3}
              style={{ width: 300, float: "left" }}
              onChange={this.onSelectionChange}
              renderInput={(params) => <TextField {...params} label="" variant="outlined" />}
            />
            { this.state.fetchInProgress ? <CircularProgress style={{float: "left", marginLeft: "20px" }}/> : <div style={{padding: "20px", float: "left"}}>{this.state.articlesToDisplay.length} results</div> }
          </Grid>
          </Grid>
        </Box>
        <Grid
          container
          spacing={2}
          direction="row"
          justifyContent="flex-start"
          alignItems="stretch"
        >
        {articleCards}
        </Grid>
        <Results />
      </Container>
    );
  }
  
  state = {
    selection: [],
    functions: [],
    articlesToDisplay: []
  };

  componentDidMount() {
    // connect to petal-api to fetch articles list.
    fetch('https://ardwrgr7s5.execute-api.us-east-2.amazonaws.com/v1/getalllabels')
    .then(res => res.json())
    .then((data) => {

      let functions = [];
      let labels = data.Items;

      labels.forEach(label => {
        functions.push({
          id: label.level3.S.toLowerCase().split(' ').join('_'),
          level2: label.level2.S,
          level3: label.level3.S
        })
      })

      this.setState({ functions: functions })
    })
    .catch(console.log)
  }

}

export default App;

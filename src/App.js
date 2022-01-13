import React, {Component} from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/CardContent';
import CardMedia from '@material-ui/core/CardMedia';
import Link from '@material-ui/core/Link';
import Grid from '@material-ui/core/Grid';

import Container from '@material-ui/core/Container';
import Typography from '@material-ui/core/Typography';
import Box from '@material-ui/core/Box';
import TextField from '@material-ui/core/TextField';
import Autocomplete from '@material-ui/lab/Autocomplete';
import Pagination from '@material-ui/lab/Pagination';
import CircularProgress from '@material-ui/core/CircularProgress';

const useStyles = makeStyles({
  root: {
    //maxWidth: 345,
    height: '100%'
  },
  media: {
    height: 140,
  },
});

function MediaCard(props) {
  const classes = useStyles();

  return (
    <Card className={classes.root}>
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
        <Box my={4}>
          <Typography variant="h4" component="h1" gutterBottom>
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
        </Box>
        <Grid
          container
          spacing={2}
          direction="row"
          justify="flex-start"
          alignItems="stretch"
        >
        {articleCards}
        </Grid>
        <Results />
      </Container>
    )
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

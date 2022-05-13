import React, {Component, useState, useEffect} from 'react';
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
import { getAllData } from '../utils/API';
import { PrivacyDialog } from '../utils/privacyHelper';

const PREFIX = 'App';

export default function HomePage(){
    const [selection, setSelection] = useState({});
    const [articlesToDisplay, setArticlesToDisplay] = useState([]);
    const [fetchInProgress, setFetchInProgress] = useState(false);
    const [functions,setFunctions] = useState([]);

    const onSelectionChange = (event, values) =>{
        setSelection(values);
        setFetchInProgress(true);
    }
    useEffect(() => {
        let unMount = false;
        if (fetchInProgress){
            //if the selection is X'd out, just fetch original articles
            if (selection == null) {
                setArticlesToDisplay([]);
                setFetchInProgress(false);
                return () => {unMount = true;};
            }

            //querying the database by selected label
            const selection_label = selection.id
            const url = new URL('https://ardwrgr7s5.execute-api.us-east-2.amazonaws.com/v1/getarticles')
            const params = { level3: selection_label }

            getAllData(params, url).then((data) => {
                data = data.filter(object => {
                return parseFloat(object.score.S) > .3;
                });

                // sort papers by scores DESC.
                data.sort(function (a, b) {
                return parseFloat(b.score.S) - parseFloat(a.score.S);
                });

                if (!unMount){
                    setArticlesToDisplay(data);
                }
                setFetchInProgress(false);
            }).catch(console.log)
        }
        return () => {unMount = true;}
    }, [fetchInProgress]);

    // Connect to PeTaL-API to fetch articles list.
    useEffect(()=>{
        fetch('https://ardwrgr7s5.execute-api.us-east-2.amazonaws.com/v1/getalllabels')
        .then(res => res.json())
        .then((data) => {
    
          let local_functions = [];
          let labels = data.Items;
    
          labels.forEach(label => {
            local_functions.push({
              id: label.level3.S.toLowerCase().split(' ').join('_'),
              level2: label.level2.S,
              level3: label.level3.S
            })
          })
    
          setFunctions(local_functions)
        })
        .catch(console.log) 
    },[]);
    
    const articleCards = articlesToDisplay.map((article) =>
      <Grid item xs={12} key={article.SortKey.S}><MediaCard article={article} /></Grid>
    );

    return (
        <Container maxWidth="ct" sx={{ mb: 3 }}>
        <Box sx={{ mt: 3, mb: 1 }}>
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
              height: 80
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
              options={functions.sort((a, b) => -b.level2.localeCompare(a.level2))}
              groupBy={(option) => option.level2}
              blurOnSelect='touch'
              getOptionLabel={(option) => option.level3}
              sx={{
                width: 350,
                float: 'left',
                mb: 2
              }}
              onChange={onSelectionChange}
              renderInput={(params) => <TextField {...params} label="" variant="standard" />}
            />
            { fetchInProgress ? <CircularProgress sx={{float: 'left', ml: 2, mb: 1 }}/> : articlesToDisplay.length ? <Box sx={{ml: 2, mb: 1, float: 'left'}}>{articlesToDisplay.length} results</Box> : ''}
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
        { !articlesToDisplay.length ? <Typography sx={{ mt: 3 }} color="text.secondary">
        Select an action from the dropdown to display a list of papers ranked by relevance to the selected action. Relevance scores for paper, action pairs were generated using a SciBERT-based multi-label text classifier fine-tuned on a small ground-truth dataset.</Typography> : ''}
        <PrivacyDialog />
      </Container>
    )
}

function MediaCard(props) {

    return (
      <Card sx={{ height: '100%', bgcolor: 'grey.100' }}>
        <CardContent>
          <Typography gutterBottom variant="h5" component="h2">
            <Link
              color="success.dark"
              target="_blank"
              rel="noopener noreferrer"
              href={props.article.url.S}
            >
              {props.article.title.S}
            </Link>
          </Typography>
          <Typography component="p" color="common.black">
            {props.article.abstract.S}
          </Typography>
          <Typography sx={{ pt: 2 }} variant="body2" color="common.black" component="p">
            Published in: {props.article.venue.S}
          </Typography>
        </CardContent>
      </Card>
    );
  }
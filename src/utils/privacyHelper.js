import { Button, Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle, Link } from "@mui/material";
import { useState } from "react";

export const PrivacyDialog = (props) => {
    const [open, setOpen] = useState(false);
    const handleClickOpen = () => setOpen(true);
    const handleClose = () => setOpen(false);

    return (
      <div>
      <Button onClick={handleClickOpen} sx={{ mt: 3 }}>Privacy Policy</Button>
      <Dialog open={open} onClose={handleClose}>
        <DialogTitle>Privacy Policy</DialogTitle>
        <DialogContent>
          <DialogContentText>
            We use a tool called “Google Analytics” to collect information about use of this site. Google Analytics collects information such as how often users visit this site, what pages they visit when they do so, and what other sites they used prior to coming to this site. We use the information we get from Google Analytics only to improve this site. Google Analytics collects only the IP address assigned to you on the date you visit this site, rather than your name or other identifying information. We do not combine the information collected through the use of Google Analytics with personally identifiable information. Although Google Analytics plants a permanent cookie on your web browser to identify you as a unique user the next time you visit this site, the cookie cannot be used by anyone but Google. Google’s ability to use and share information collected by Google Analytics about your visits to this site is restricted by the <Link href="https://www.google.com/analytics/terms/" target="_blank" rel="noreferrer">Google Analytics Terms of Use</Link> and the <Link href="https://policies.google.com/privacy" target="_blank" rel="noreferrer">Google Privacy Policy</Link>. You can prevent Google Analytics from recognizing you on return visits to this site by <Link href="http://www.usa.gov/optout_instructions.shtml" target="_blank" rel="noreferrer">disabling cookies</Link> in your browser.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClose}>Close</Button>
        </DialogActions>
      </Dialog>
      </div>
    );
  }